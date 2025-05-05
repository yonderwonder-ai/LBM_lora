import importlib
import logging
import re
import time
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from ..data.mappers import MapperWrapper
from ..models.base.base_model import BaseModel
from .training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)


class TrainingPipeline(pl.LightningModule):
    """
    Main Training Pipeline class for ClipDrop.

    Args:

        model (BaseModel): The model to train
        pipeline_config (TrainingConfig): The configuration for the training pipeline
        cuda_mapper (Optional[MapperWrapper]): An instance of the MapperWrapper class to apply mappers that require GPU.
            This will be called in the [`on_after_batch_transfer` hook](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.DataHooks.html#lightning.pytorch.core.hooks.DataHooks.on_after_batch_transfer).
        verbose (bool): Whether to print logs in the console. Default is False.
    """

    def __init__(
        self,
        model: BaseModel,
        pipeline_config: TrainingConfig,
        cuda_mappers: Optional[MapperWrapper] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.pipeline_config = pipeline_config
        self.cuda_mappers = cuda_mappers
        self.log_samples_model_kwargs = pipeline_config.log_samples_model_kwargs

        # save hyperparameters.
        self.save_hyperparameters(ignore="model")
        self.save_hyperparameters({"model_config": model.config.to_dict()})

        # logger.
        self.verbose = verbose

        # setup logging.
        log_keys = pipeline_config.log_keys

        if isinstance(log_keys, str):
            log_keys = [log_keys]

        if log_keys is None:
            log_keys = []

        self.log_keys = log_keys
        self._optimizer_update_steps_counter = 0
        self._optimizer_idx = 0

    def on_fit_start(self) -> None:
        self.model.on_fit_start(device=self.device)
        if self.global_rank == 0:
            self.timer = time.perf_counter()

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.cuda_mappers is not None:
            return self.cuda_mappers(batch)
        return batch

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        if self.global_rank == 0:
            logging.debug("on_train_batch_end")
        self.model.on_train_batch_end(batch)

        average_time_frequency = 10
        if self.global_rank == 0 and batch_idx % average_time_frequency == 0:
            delta = time.perf_counter() - self.timer
            logging.info(
                f"Average time per batch {batch_idx} took {delta / (batch_idx + 1)} seconds"
            )

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Setup optimizers and learning rate schedulers.
        """
        optimizers = []
        for i in range(len(self.pipeline_config.optimizers_name)):
            lr = self.pipeline_config.learning_rates[i]
            param_list = []
            n_params = 0
            param_list_ = {"params": []}
            for name, param in self.model.named_parameters():
                for regex in self.pipeline_config.trainable_params[i]:
                    pattern = re.compile(regex)
                    if re.match(pattern, name):
                        if param.requires_grad:
                            param_list_["params"].append(param)
                            n_params += param.numel()

            param_list.append(param_list_)

            logging.info(
                f"Number of trainable parameters for optimizer {i}: {n_params}"
            )

            optimizer_cls = getattr(
                importlib.import_module("torch.optim"),
                self.pipeline_config.optimizers_name[i],
            )
            optimizer = optimizer_cls(
                param_list, lr=lr, **self.pipeline_config.optimizers_kwargs[i]
            )
            optimizers.append(optimizer)

        if len(optimizers) > 1:
            self.automatic_optimization = False

        self.optims = optimizers
        schedulers_config = self.configure_lr_schedulers()

        for name, param in self.model.named_parameters():
            set_grad_false = True
            for regexes in self.pipeline_config.trainable_params:
                for regex in regexes:
                    pattern = re.compile(regex)
                    if re.match(pattern, name):
                        if param.requires_grad:
                            set_grad_false = False
            if set_grad_false:
                param.requires_grad = False

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logging.info(f"Number of trainable parameters: {num_trainable_params}")

        schedulers_config = self.configure_lr_schedulers()

        if schedulers_config is None:
            return optimizers

        return optimizers, [
            schedulers_config_ for schedulers_config_ in schedulers_config
        ]

    def configure_lr_schedulers(self) -> List[Dict[str, Any]]:
        schedulers_config = []
        for i in range(len(self.pipeline_config.lr_schedulers_name)):
            if self.pipeline_config.lr_schedulers_name[i] is None:
                scheduler = None
                schedulers_config.append(scheduler)
            else:
                scheduler_cls = getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.pipeline_config.lr_schedulers_name[i],
                )
                scheduler = scheduler_cls(
                    self.optims[i],
                    **self.pipeline_config.lr_schedulers_kwargs[i],
                )
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": self.pipeline_config.lr_schedulers_interval[i],
                    "monitor": "val_loss",
                    "frequency": self.pipeline_config.lr_schedulers_frequency[i],
                }
                schedulers_config.append(lr_scheduler_config)

        if all([scheduler is None for scheduler in schedulers_config]):
            return None

        return schedulers_config

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> dict:
        if self.automatic_optimization:
            model_output = self.model(train_batch)
            loss = model_output["loss"]
            logging.info(f"loss: {loss}")
            return {
                "loss": loss,
                "batch_idx": batch_idx,
            }
        # manual optim for multiple optimizers
        else:
            self._optimizer_update_steps_counter += 1
            print(
                "self._optimizer_update_steps_counter",
                self._optimizer_update_steps_counter,
            )
            num_optimizer_update_steps = self.pipeline_config.optimizers_update_steps[
                self._optimizer_idx
            ]
            print("num_optimizer_update_steps", num_optimizer_update_steps)
            optim_idx = self._optimizer_idx
            if self._optimizer_update_steps_counter >= num_optimizer_update_steps:
                self._optimizer_idx = (self._optimizer_idx + 1) % len(self.optimizers())
                self._optimizer_update_steps_counter = 0
            optimizers = self.optimizers()
            opt = optimizers[optim_idx]
            outputs = {"batch_idx": batch_idx}
            self.toggle_optimizer(opt)
            opt.zero_grad()
            model_output = self.model(train_batch, step=optim_idx, batch_idx=batch_idx)
            loss = model_output["loss"][optim_idx]
            if self.global_rank == 0:
                logging.info(
                    f"batch_idx:{batch_idx}, loss for optimizer {optim_idx}: {loss}"
                )
            outputs[f"loss_optimizer_{optim_idx}"] = loss
            self.manual_backward(loss)
            opt.step()
            self.untoggle_optimizer(opt)
        return outputs

    def validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
        loss = self.model(val_batch, device=self.device)["loss"]

        metrics = self.model.compute_metrics(val_batch)

        return {"loss": loss, "metrics": metrics}

    def log_samples(self, batch: Dict[str, Any]):
        logging.debug("log_samples")
        logs = self.model.log_samples(
            batch,
            **self.log_samples_model_kwargs,
        )

        if logs is not None:
            N = min([logs[keys].shape[0] for keys in logs])
        else:
            N = 0

        # Log inputs
        if self.log_keys is not None:
            for key in self.log_keys:
                if key in batch:
                    if N > 0:
                        logs[key] = batch[key][:N]
                    else:
                        logs[key] = batch[key]

        return logs
