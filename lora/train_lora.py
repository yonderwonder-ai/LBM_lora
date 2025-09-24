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
from typing import List, Optional

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
    rank: int,
    alpha: int,
    target_modules: List[str],
    trainable_lora_modules: List[str],
):
    """
    Injects LoRA layers and freezes/unfreezes weights for selective training.

    Args:
        model (LBMModel): The LBM model to modify.
        rank (int): The rank of the LoRA matrices.
        alpha (int): The alpha scaling factor for LoRA.
        target_modules (List[str]): Patterns for layers to inject LoRA into.
        trainable_lora_modules (List[str]): Patterns for which injected LoRA
                                            layers to train.
    """
    denoiser = model.denoiser

    # Inject LoRA layers first
    for name, module in list(denoiser.named_modules()):
        if not any(target in name for target in target_modules):
            continue

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = denoiser.get_submodule(parent_name)

            if isinstance(module, nn.Linear):
                new_module = LoRALinear(module, rank, alpha)
            elif isinstance(module, nn.Conv2d):
                new_module = LoRAConv2d(module, rank, alpha)
            else:
                continue

            setattr(parent_module, child_name, new_module)

    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the parameters of specified LoRA modules
    trainable_param_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            if any(
                trainable_target in name for trainable_target in trainable_lora_modules
            ):
                for param in module.lora_down.parameters():
                    param.requires_grad = True
                    trainable_param_count += param.numel()
                for param in module.lora_up.parameters():
                    param.requires_grad = True
                    trainable_param_count += param.numel()

    logging.info(
        f"Froze all model weights. Unfroze {trainable_param_count:,} parameters for selective LoRA training."
    )


def get_model(
    base_model_path: str,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: List[str],
    trainable_lora_modules: List[str],
    lora_weights_path: Optional[str] = None,
):
    """
    Loads a base LBM model and injects LoRA layers for fine-tuning.
    Optionally loads pre-existing LoRA weights to continue training.
    """
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

    logging.info(f"Injecting LoRA layers with rank={lora_rank}, alpha={lora_alpha}")
    add_lora_to_model(
        model, lora_rank, lora_alpha, lora_target_modules, trainable_lora_modules
    )

    if lora_weights_path:
        logging.info(f"Loading existing LoRA weights from: {lora_weights_path}")
        from safetensors.torch import load_file

        lora_state_dict = load_file(lora_weights_path, device="cpu")
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
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_target_modules: List[str] = ["attn1", "ff.net"],
    trainable_lora_modules: Optional[List[str]] = None,
    lora_weights_path: Optional[str] = None,
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
        trainable_lora_modules = lora_target_modules

    model = get_model(
        base_model_path=base_model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
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
        trainable_params=[".*"],  # This will select all params with requires_grad=True
        optimizer_name=optimizer,
        log_samples_model_kwargs={"num_steps": num_steps},
    )

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)
    pipeline.save_hyperparameters(
        {
            "lora_config": {
                "rank": lora_rank,
                "alpha": lora_alpha,
                "target_modules": lora_target_modules,
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
