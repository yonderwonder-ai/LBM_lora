"""
LBM LoRA Fine-Tuning Script

This script fine-tunes LBM models using Low-Rank Adaptation (LoRA).
It adapts the original relighting training script to support LoRA by:
- Loading a pre-initialized LBM base model.
- Injecting LoRA layers into the denoiser.
- Freezing all non-LoRA parameters for efficient training.
"""

import datetime
import logging
import os
import random
import re
import shutil
import sys
from typing import List, Optional, Dict, Any

import braceexpand
import fire
import torch
import torch.nn as nn
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file

# Add LBM source path
LBM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LBM_ROOT, "src"))
sys.path.insert(0, os.path.join(LBM_ROOT, "lora"))


from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.lbm import LBMModel
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger
from lora_layers import LoRAConv2d, LoRALinear
from lbm_utils import load_lbm_model_from_checkpoint


def add_lora_to_model(
    model: LBMModel,
    lora_config_map: Dict[str, Dict[str, Any]],
    trainable_lora_modules: List[str],
):
    """
    Injects LoRA layers and freezes/unfreezes weights for selective training.

    Args:
        model (LBMModel): The LBM model to modify.
        lora_config_map (Dict): A map from module name to its LoRA config (rank, alpha).
        trainable_lora_modules (List[str]): Patterns for which injected LoRA
                                            layers to train.
    """
    denoiser = model.denoiser

    # Inject LoRA layers based on the config map
    for module_name, config in lora_config_map.items():
        try:
            # Remove "denoiser." prefix to use get_submodule, as it operates within the denoiser
            submodule_path = module_name.replace("denoiser.", "", 1)

            if "." in submodule_path:
                parent_name, child_name = submodule_path.rsplit(".", 1)
                parent_module = denoiser.get_submodule(parent_name)
            else:
                # This handles modules that are direct children of the denoiser (e.g., 'conv_in')
                parent_module = denoiser
                child_name = submodule_path

            module_to_replace = getattr(parent_module, child_name)

        except (AttributeError, ValueError):
            logging.warning(
                f"Could not find module {module_name} in the model. Skipping injection."
            )
            continue

        rank = config["rank"]
        alpha = config["alpha"]

        if isinstance(module_to_replace, nn.Linear):
            new_module = LoRALinear(module_to_replace, rank, alpha)
        elif isinstance(module_to_replace, nn.Conv2d):
            new_module = LoRAConv2d(module_to_replace, rank, alpha)
        else:
            continue

        setattr(parent_module, child_name, new_module)

    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the parameters of specified LoRA modules
    trainable_param_count = 0
    trainable_params_set = set()

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            if any(
                trainable_target in name for trainable_target in trainable_lora_modules
            ):
                for param in module.lora_down.parameters():
                    if param not in trainable_params_set:
                        param.requires_grad = True
                        trainable_params_set.add(param)
                        trainable_param_count += param.numel()
                for param in module.lora_up.parameters():
                    if param not in trainable_params_set:
                        param.requires_grad = True
                        trainable_params_set.add(param)
                        trainable_param_count += param.numel()

    logging.info(
        f"Froze all model weights. Unfroze {trainable_param_count:,} unique parameters for selective LoRA training."
    )


def get_model(
    base_model_path: str,
    trainable_lora_modules: List[str],
    lora_weights_path: str,
):
    """
    Loads a base LBM model, injects LoRA layers based on a pre-trained LoRA
    weights file, and loads those weights for fine-tuning.
    """
    logging.info(f"Loading existing LoRA weights from: {lora_weights_path}")
    lora_state_dict = load_file(lora_weights_path, device="cpu")

    # --- Infer LoRA configuration from the state dict ---
    lora_config_map = {}
    grouped_lora_weights = {}

    # Group up, down, alpha weights by module name
    for key, tensor in lora_state_dict.items():
        match = re.search(r"(denoiser\..*)\.(lora_up|lora_down|alpha)", key)
        if not match:
            continue

        module_name = match.group(1)
        lora_type = match.group(2)

        if module_name not in grouped_lora_weights:
            grouped_lora_weights[module_name] = {}

        if lora_type in ["lora_up", "lora_down"]:
            grouped_lora_weights[module_name][lora_type] = tensor
        elif lora_type == "alpha":
            grouped_lora_weights[module_name][lora_type] = tensor

    # Infer rank and alpha from the grouped weights
    for module_name, weights in grouped_lora_weights.items():
        if "lora_down" not in weights:
            logging.warning(
                f"Incomplete LoRA module {module_name} found (missing lora_down). Skipping."
            )
            continue

        # Rank is the output dimension of the 'down' projection
        rank = weights["lora_down"].shape[0]

        # Alpha is stored directly, default to rank if not present
        alpha = weights.get("alpha", torch.tensor(float(rank))).item()

        lora_config_map[module_name] = {"rank": rank, "alpha": int(alpha)}

    logging.info(
        f"Inferred LoRA configuration for {len(lora_config_map)} modules from state dict."
    )
    # --- End of inference ---

    logging.info(f"Loading base model from: {base_model_path}")
    model = load_lbm_model_from_checkpoint(
        base_model_path, device="cpu", torch_dtype=torch.bfloat16
    )

    # Force source and target keys to match the data pipeline
    model.config.source_key = "source_image"
    model.source_key = "source_image"
    model.config.target_key = "target_image"
    model.target_key = "target_image"
    logging.info(
        f"Set model source_key to '{model.source_key}' and target_key to '{model.target_key}'"
    )

    logging.info(f"Injecting LoRA layers with inferred ranks and alphas...")
    add_lora_to_model(model, lora_config_map, trainable_lora_modules)

    # Now that requires_grad is set, we can load the state dict.
    # The optimizer will be configured later by the Trainer.
    incompatible_keys = model.load_state_dict(lora_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        logging.info(
            "LoRA weights loaded. Some keys were missing from the checkpoint, which is expected."
        )
    if incompatible_keys.unexpected_keys:
        logging.warning(
            f"Warning: Unexpected keys found in LoRA checkpoint: {incompatible_keys.unexpected_keys}"
        )

    return model.to(torch.bfloat16)


def get_filter_mappers():
    """
    Data pipeline for relighting training - no mask processing needed.
    """
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["source.jpg", "target.jpg"])),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "source.jpg": "source_image",
                            "target.jpg": "target_image",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="source_image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="target_image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="source_image")),
                RescaleMapper(RescaleMapperConfig(key="target_image")),
            ],
        ),
    ]
    return filters_mappers


def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
):
    # TRAIN
    train_filters_mappers = get_filter_mappers()
    train_shards_path_or_urls_unbraced = [
        s for url in train_shards for s in braceexpand.braceexpand(url)
    ]
    random.shuffle(train_shards_path_or_urls_unbraced)

    train_data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    # VALIDATION
    validation_filters_mappers = get_filter_mappers()
    validation_shards_path_or_urls_unbraced = [
        s for url in validation_shards for s in braceexpand.braceexpand(url)
    ]
    
    validation_data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(validation_shards_path_or_urls_unbraced)),
    )

    return DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )


def main(
    train_shards: List[str],
    validation_shards: List[str],
    base_model_path: str,
    trainable_lora_modules: Optional[List[str]] = None,
    lora_weights_path: str = None,
    wandb_project: str = "lbm-lora-relighting",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 1e-4,
    optimizer: str = "AdamW",
    save_ckpt_path: str = "./checkpoints_lora",
    log_interval: int = 100,
    max_epochs: int = 100,
    save_interval: int = 1000,
    path_config: str = None,
    config_yaml: dict = None,
):
    # If trainable_lora_modules isn't specified, default to training all injected modules
    if trainable_lora_modules is None:
        trainable_lora_modules = []  # Will be inferred, but needs to be a list

    model = get_model(
        base_model_path=base_model_path,
        trainable_lora_modules=trainable_lora_modules,
        lora_weights_path=lora_weights_path,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
    )

    # Train only the parameters that were unfrozen in add_lora_to_model
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        log_keys=["source_image", "target_image"],
        trainable_params=[".*lora.*"],  # Now we can just train all params with 'lora' in their name
        optimizer_name=optimizer,
        log_samples_model_kwargs={"num_steps": num_steps},
    )

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)
    pipeline.save_hyperparameters(
        {
            "lora_config": {
                "trainable_modules": trainable_lora_modules,
            },
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
        }
    )

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-LBM-LoRA-Relighting"
    )
    run_name = training_signature

    if wandb_project and wandb_project != "null":
        logger_instance = loggers.WandbLogger(
            project=wandb_project, name=run_name, save_dir=save_ckpt_path
        )
    else:
        logger_instance = loggers.TensorBoardLogger(
            save_dir=save_ckpt_path, name="lora_relighting_test"
        )
    
    callbacks_list = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=save_ckpt_path,
            every_n_train_steps=save_interval,
            save_last=True,
        ),
    ]
    if wandb_project and wandb_project != "null":
        callbacks_list.insert(0, WandbSampleLogger(log_batch_freq=log_interval))

    trainer = Trainer(
        accelerator="gpu",
        devices=int(os.environ.get("SLURM_NPROCS", 1)),
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy="auto",
        logger=logger_instance,
        callbacks=callbacks_list,
        precision="bf16-mixed",
        max_epochs=max_epochs,
    )

    trainer.fit(pipeline, data_module)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(f"Running main with config: {config}")
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main_from_config)
