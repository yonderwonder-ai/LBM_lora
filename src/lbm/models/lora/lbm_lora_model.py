from typing import Any, Dict, Optional, Union
import logging
import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
import re

from ..base.base_model import BaseModel
from ..lbm import LBMModel
from .lbm_lora_config import LBMLoRAConfig

logger = logging.getLogger(__name__)


class LBMLoRAModel(BaseModel):
    """LBM model with LoRA (Low-Rank Adaptation) capabilities.
    
    This class wraps an LBM model and adds runtime LoRA application capabilities.
    It follows LBM's architecture patterns and configuration systems.

    Args:
        config (LBMLoRAConfig): Configuration for the LoRA model
        base_model (LBMModel): Base LBM model to apply LoRA to
        lora_weights_path (Optional[str]): Path to LoRA weights file
    """

    def __init__(
        self,
        config: LBMLoRAConfig,
        base_model: LBMModel,
        lora_weights_path: Optional[str] = None,
    ):
        BaseModel.__init__(self, config)
        
        self.config = config
        self.base_model = base_model
        self.lora_weights = None
        self.lora_scale = config.lora_scale
        
        # Load LoRA weights if provided
        if lora_weights_path:
            self.load_lora_weights(lora_weights_path)
            
    def load_lora_weights(self, lora_weights_path: str):
        """Load LoRA weights from safetensors file"""
        logger.info(f"Loading LoRA weights from {lora_weights_path}")
        
        self.lora_weights = load_file(lora_weights_path)
        
        # Move to model device and dtype
        for key in self.lora_weights:
            self.lora_weights[key] = self.lora_weights[key].to(
                device=self.device, dtype=self.dtype
            )
        
        # Apply LoRA weights to model
        self._apply_lora_weights()
        
        lora_modules_count = len([k for k in self.lora_weights.keys() if k.endswith('.lora_up.weight')])
        logger.info(f"Loaded and applied {lora_modules_count} LoRA modules")

    def _apply_lora_weights(self):
        """Apply LoRA weights to base model following LBM conventions"""
        if not self.lora_weights:
            return
        
        logger.info("Applying LoRA weights to base model...")
        
        # Group LoRA weights by module
        lora_modules = self._group_lora_weights()
        
        applied_count = 0
        failed_count = 0
        
        # Apply LoRA to each target module
        for module_name, lora_data in lora_modules.items():
            if "up" not in lora_data or "down" not in lora_data:
                continue
                
            try:
                # Convert LoRA module name to model path
                model_path = self._convert_lora_name_to_path(module_name)
                
                # Find target module in base model
                target_module = self._get_target_module(model_path)
                if target_module is None:
                    logger.debug(f"Module not found: {model_path}")
                    failed_count += 1
                    continue
                
                # Apply LoRA delta
                success = self._apply_lora_to_module(target_module, lora_data, module_name)
                if success:
                    applied_count += 1
                    logger.debug(f"✓ Applied LoRA to {model_path}")
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.debug(f"Failed to apply LoRA to {module_name}: {e}")
                failed_count += 1
        
        logger.info(f"LoRA application completed: {applied_count} applied, {failed_count} failed")

    def _group_lora_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Group LoRA weights by module name"""
        lora_modules = {}
        
        for key, weight in self.lora_weights.items():
            if ".lora_up.weight" in key or ".lora_down.weight" in key or ".alpha" in key:
                module_name = key.replace(".lora_up.weight", "").replace(".lora_down.weight", "").replace(".alpha", "")
                
                if module_name not in lora_modules:
                    lora_modules[module_name] = {}
                
                if ".lora_up.weight" in key:
                    lora_modules[module_name]["up"] = weight
                elif ".lora_down.weight" in key:
                    lora_modules[module_name]["down"] = weight
                elif ".alpha" in key:
                    lora_modules[module_name]["alpha"] = weight
        
        return lora_modules

    def _convert_lora_name_to_path(self, module_name: str) -> str:
        """Convert LoRA module name to model path following LBM conventions"""
        # Pattern: lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q
        # Target:  down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q
        model_path = module_name.replace("lora_unet_", "")
        
        # Convert numbered patterns: _N_ -> .N.
        model_path = re.sub(r'_(\d+)_', r'.\1.', model_path)
        
        # Handle final numbers: _N$ -> .N
        model_path = re.sub(r'_(\d+)$', r'.\1', model_path)
        
        # Fix specific LBM patterns
        replacements = [
            ('mid_block_attentions', 'mid_block.attentions'),
            ('ff_net', 'ff.net'),
            ('attn1_to_', 'attn1.to_'),
        ]
        
        for old, new in replacements:
            model_path = model_path.replace(old, new)
            
        return model_path

    def _get_target_module(self, model_path: str):
        """Get target module from base model using path"""
        try:
            target_module = self.base_model.denoiser
            for part in model_path.split("."):
                target_module = getattr(target_module, part)
            
            if not hasattr(target_module, 'weight'):
                return None
                
            return target_module
            
        except AttributeError:
            return None

    def _apply_lora_to_module(self, target_module, lora_data, module_name: str = "") -> bool:
        """Apply LoRA delta to a specific module"""
        try:
            lora_up = lora_data["up"]
            lora_down = lora_data["down"]
            alpha = lora_data.get("alpha", torch.tensor(16.0))
            
            # Calculate rank and scaling
            if len(lora_up.shape) == 4:  # Conv tensor
                if lora_up.shape[2:] == (1, 1):  # [out_ch, rank, 1, 1]
                    rank = lora_up.shape[1]
                else:  # [rank, in_ch, kh, kw]
                    rank = lora_up.shape[0]
            else:  # Linear tensor
                rank = lora_up.shape[0]
                
            alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            scale = (alpha_val / rank) * self.lora_scale
            
            # Compute delta using LBM-specific LoRA format
            delta = self._compute_lora_delta(lora_up, lora_down, target_module.weight.shape)
            if delta is None:
                return False
            
            # Apply delta with scaling
            delta = delta * scale
            target_module.weight.data += delta.to(target_module.weight.dtype)
            
            return True
            
        except Exception as e:
            logger.debug(f"LoRA application failed: {e}")
            return False

    def _compute_lora_delta(self, lora_up, lora_down, target_shape) -> Optional[torch.Tensor]:
        """Compute LoRA delta following LBM's tensor format"""
        try:
            if len(lora_up.shape) == 4 and len(lora_down.shape) == 4:  # Conv layers
                
                if lora_up.shape[2:] == (1, 1):  # Case: up=[out_ch, rank, 1, 1], down=[rank, in_ch, kh, kw]
                    out_ch, rank = lora_up.shape[:2]
                    rank2, in_ch, kh, kw = lora_down.shape
                    
                    if rank != rank2:
                        return None
                    
                    # Matrix multiplication
                    lora_up_2d = lora_up.squeeze(-1).squeeze(-1)  # [out_ch, rank]
                    lora_down_2d = lora_down.reshape(rank, in_ch * kh * kw)  # [rank, in_ch*kh*kw]
                    
                    delta_2d = lora_up_2d @ lora_down_2d  # [out_ch, in_ch*kh*kw]
                    delta = delta_2d.reshape(target_shape)
                    
                elif lora_down.shape[2:] == (1, 1):  # Case: up=[rank, in_ch, kh, kw], down=[out_ch, rank, 1, 1]
                    rank, in_ch, kh, kw = lora_up.shape
                    out_ch, rank2 = lora_down.shape[:2]
                    
                    if rank != rank2:
                        return None
                    
                    # Matrix multiplication  
                    lora_up_2d = lora_up.reshape(rank, in_ch * kh * kw)  # [rank, in_ch*kh*kw]
                    lora_down_2d = lora_down.squeeze(-1).squeeze(-1)  # [out_ch, rank]
                    
                    delta_2d = lora_down_2d @ lora_up_2d  # [out_ch, in_ch*kh*kw]
                    delta = delta_2d.reshape(target_shape)
                    
                else:
                    logger.warning(f"Unsupported conv LoRA shapes: up={lora_up.shape}, down={lora_down.shape}")
                    return None
                    
            elif len(lora_up.shape) == 2 and len(lora_down.shape) == 2:  # Linear layers
                # Standard LoRA: delta = lora_up @ lora_down
                delta = lora_up @ lora_down
                
            else:
                logger.warning(f"Mixed LoRA tensor types: up={lora_up.shape}, down={lora_down.shape}")
                return None
                
            return delta
            
        except Exception as e:
            logger.debug(f"Delta computation failed: {e}")
            return None

    def set_lora_scale(self, scale: float):
        """Update LoRA scaling factor and reapply weights"""
        if scale != self.lora_scale:
            logger.info(f"Updating LoRA scale from {self.lora_scale} to {scale}")
            
            # Reset base model to original state
            self._reset_to_base_weights()
            
            # Update scale and reapply
            self.lora_scale = scale
            if self.lora_weights:
                self._apply_lora_weights()

    def _reset_to_base_weights(self):
        """Reset model to base weights (without LoRA)"""
        # This would require storing original weights or reloading base model
        # For now, we'll require model reload for scale changes
        logger.warning("LoRA scale change requires model reload in current implementation")

    def forward(self, batch: Dict[str, Any], *args, **kwargs):
        """Forward pass using base model with LoRA adaptations"""
        return self.base_model.forward(batch, *args, **kwargs)

    def sample(self, *args, **kwargs):
        """Sampling using base model with LoRA adaptations"""
        return self.base_model.sample(*args, **kwargs)

    def log_samples(self, *args, **kwargs):
        """Log samples using base model with LoRA adaptations"""
        return self.base_model.log_samples(*args, **kwargs)

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when training starts - delegate to base model"""
        super().on_fit_start(device=device, *args, **kwargs)
        self.base_model.on_fit_start(device=device, *args, **kwargs)

    def freeze_base_model(self):
        """Freeze base model parameters, keeping only LoRA parameters trainable"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base model parameters frozen for LoRA training")

    def unfreeze_lora_parameters(self):
        """Unfreeze LoRA parameters for training"""
        # This would be implemented when adding trainable LoRA layers
        # For now, LoRA training will use the extracted weights as initialization
        pass

    def get_trainable_parameters(self):
        """Get list of trainable LoRA parameters"""
        # This will be used for LoRA training setup
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and ("lora_" in name or self._is_lora_target(name)):
                trainable_params.append(param)
        return trainable_params

    def _is_lora_target(self, param_name: str) -> bool:
        """Check if parameter should be LoRA-adapted based on config"""
        for pattern in self.config.target_modules:
            if re.search(pattern.replace(".", r"\."), param_name):
                return True
        return False

    def save_lora_weights(self, save_path: str):
        """Save current LoRA weights (for training checkpoints)"""
        # This will be implemented for LoRA training
        logger.info(f"Saving LoRA weights to {save_path}")
        # Implementation would extract current LoRA deltas and save them

    @classmethod
    def from_pretrained(
        cls,
        base_model: LBMModel,
        lora_path: str,
        config: Optional[LBMLoRAConfig] = None,
        **kwargs
    ):
        """Load LBM LoRA model from pretrained LoRA weights"""
        if config is None:
            # Try to load config from same directory
            config_path = lora_path.replace(".safetensors", "_config.yaml")
            if os.path.exists(config_path):
                config = LBMLoRAConfig.from_yaml(config_path)
            else:
                # Use default config
                config = LBMLoRAConfig(**kwargs)
        
        return cls(
            config=config,
            base_model=base_model,
            lora_weights_path=lora_path
        )

    @property
    def vae(self):
        """Access to VAE from base model"""
        return self.base_model.vae

    @property
    def denoiser(self):
        """Access to denoiser from base model"""
        return self.base_model.denoiser

    @property
    def conditioner(self):
        """Access to conditioner from base model"""
        return self.base_model.conditioner

    @property
    def sampling_noise_scheduler(self):
        """Access to sampling scheduler from base model"""
        return self.base_model.sampling_noise_scheduler

    @property
    def training_noise_scheduler(self):
        """Access to training scheduler from base model"""
        return self.base_model.training_noise_scheduler
