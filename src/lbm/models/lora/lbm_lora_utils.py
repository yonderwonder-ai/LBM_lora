"""
LBM LoRA Utilities

This module provides utilities for creating, managing, and applying LoRA adaptations 
to LBM models following LBM's architecture patterns and conventions.
"""

from typing import List, Optional, Dict, Any
import torch
import logging
import re
from safetensors.torch import save_file

from ..lbm import LBMModel
from .lbm_lora_config import LBMLoRAConfig
from .lbm_lora_model import LBMLoRAModel

logger = logging.getLogger(__name__)


def create_lbm_lora_model(
    base_model: LBMModel,
    lora_config: Optional[LBMLoRAConfig] = None,
    lora_weights_path: Optional[str] = None,
    **config_kwargs
) -> LBMLoRAModel:
    """
    Create LBM LoRA model following LBM conventions.
    
    Args:
        base_model: Base LBM model to adapt
        lora_config: LoRA configuration
        lora_weights_path: Path to LoRA weights
        **config_kwargs: Additional config parameters
    
    Returns:
        LBMLoRAModel: LoRA-adapted model
    """
    if lora_config is None:
        lora_config = LBMLoRAConfig(**config_kwargs)
    
    return LBMLoRAModel(
        config=lora_config,
        base_model=base_model,
        lora_weights_path=lora_weights_path
    )


def get_lora_target_modules(coverage: str = "attention") -> List[str]:
    """
    Get target module patterns for LoRA adaptation following LBM architecture.
    
    Args:
        coverage: Coverage level ("attention", "extended", "full")
    
    Returns:
        List of module name patterns
    """
    if coverage == "attention":
        return [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        ]
    elif coverage == "extended":
        return [
            # Attention layers
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            # Feed-forward layers
            "ff.net.0.proj", "ff.net.2",
        ]
    elif coverage == "full":
        return [
            # Attention layers
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            # Feed-forward layers
            "ff.net.0.proj", "ff.net.2",
            # Convolutional layers
            "conv1", "conv2", "conv_shortcut",
            # Time embedding
            "time_emb_proj",
            # Input/Output convolutions
            "conv_in", "conv_out",
            # Sampling layers
            "conv", "downsamplers.0.conv", "upsamplers.0.conv",
        ]
    else:
        raise ValueError(f"Unknown coverage: {coverage}")


def extract_lora_from_models(
    base_model: LBMModel,
    fine_tuned_model: LBMModel,
    config: LBMLoRAConfig,
    save_path: Optional[str] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from fine-tuned model using LBM conventions.
    
    Args:
        base_model: Base LBM model
        fine_tuned_model: Fine-tuned LBM model  
        config: LoRA configuration
        save_path: Path to save extracted LoRA weights
        device: Device for computation
    
    Returns:
        Dictionary of LoRA weights
    """
    logger.info("Extracting LoRA weights from LBM models...")
    
    # Get target modules based on config
    target_modules = config.target_modules
    
    # Find matching modules in both models
    base_modules = _find_target_modules(base_model.denoiser, target_modules)
    tuned_modules = _find_target_modules(fine_tuned_model.denoiser, target_modules)
    
    # Extract LoRA weights via SVD
    lora_weights = {}
    extracted_count = 0
    
    for module_path in base_modules:
        if module_path not in tuned_modules:
            continue
            
        base_weight = base_modules[module_path].weight.to(device)
        tuned_weight = tuned_modules[module_path].weight.to(device)
        
        # Calculate difference
        diff = tuned_weight - base_weight
        
        # Check if difference is significant
        if torch.max(torch.abs(diff)).item() < 0.001:
            continue
        
        # Perform SVD and create LoRA weights
        lora_up, lora_down, alpha = _svd_to_lora(
            diff, 
            rank=config.conv_rank if len(diff.shape) == 4 else config.rank,
            alpha=config.conv_alpha if len(diff.shape) == 4 else config.lora_alpha
        )
        
        if lora_up is not None and lora_down is not None:
            # Convert module path to LoRA name
            lora_name = _module_path_to_lora_name(module_path)
            
            lora_weights[f"{lora_name}.lora_up.weight"] = lora_up
            lora_weights[f"{lora_name}.lora_down.weight"] = lora_down
            lora_weights[f"{lora_name}.alpha"] = alpha
            extracted_count += 1
    
    logger.info(f"Extracted LoRA from {extracted_count} modules")
    
    # Save if path provided
    if save_path:
        save_file(lora_weights, save_path)
        logger.info(f"LoRA weights saved to {save_path}")
    
    return lora_weights


def _find_target_modules(model: torch.nn.Module, target_patterns: List[str]) -> Dict[str, torch.nn.Module]:
    """Find modules matching target patterns"""
    target_modules = {}
    
    for name, module in model.named_modules():
        for pattern in target_patterns:
            # Convert pattern to regex
            regex_pattern = pattern.replace(".", r"\.").replace("*", r".*") + "$"
            if re.search(regex_pattern, name):
                if hasattr(module, 'weight'):
                    target_modules[name] = module
                break
    
    return target_modules


def _module_path_to_lora_name(module_path: str) -> str:
    """Convert module path to LoRA naming convention"""
    # Convert: down_blocks.1.attentions.0.attn1.to_q -> lora_unet_down_blocks_1_attentions_0_attn1_to_q
    lora_name = module_path.replace(".", "_")
    return f"lora_unet_{lora_name}"


def _svd_to_lora(diff: torch.Tensor, rank: int, alpha: float) -> tuple:
    """Convert weight difference to LoRA components via SVD"""
    try:
        # Prepare matrix for SVD
        if len(diff.shape) == 4:  # Conv layer
            is_3x3_conv = diff.shape[2:] != (1, 1)
            if is_3x3_conv:
                # For 3x3 conv: flatten spatial dimensions
                diff_2d = diff.flatten(start_dim=1)  # [out_ch, in_ch*kh*kw]
            else:
                # For 1x1 conv: treat as linear
                diff_2d = diff.squeeze(-1).squeeze(-1)  # [out_ch, in_ch]
        else:
            # Linear layer
            diff_2d = diff
        
        # Perform SVD
        U, S, Vh = torch.linalg.svd(diff_2d.float())
        
        # Select rank components
        rank = min(rank, min(diff_2d.shape), len(S))
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]
        
        # Create LoRA matrices with singular value splitting
        s_sqrt = torch.sqrt(torch.clamp(S_k, min=0.0))
        lora_up = U_k * s_sqrt.unsqueeze(0)  # [out_dim, rank]
        lora_down = Vh_k * s_sqrt.unsqueeze(1)  # [rank, in_dim]
        
        # Reshape back for conv layers if needed
        if len(diff.shape) == 4:  # Conv layer
            out_ch, in_ch, kh, kw = diff.shape
            
            if diff.shape[2:] != (1, 1):  # 3x3 conv
                # LBM format: up=[rank, in_ch, kh, kw], down=[out_ch, rank, 1, 1]
                lora_down = lora_up.T.contiguous()  # [rank, out_ch] -> [out_ch, rank]
                lora_down = lora_down.unsqueeze(-1).unsqueeze(-1)  # [out_ch, rank, 1, 1]
                
                lora_up = Vh_k.T.contiguous()  # [in_dim, rank] -> [rank, in_dim]
                lora_up = lora_up.reshape(rank, in_ch, kh, kw)  # [rank, in_ch, kh, kw]
            else:
                # 1x1 conv: keep as [out_ch, rank, 1, 1] and [rank, in_ch, 1, 1]
                lora_up = lora_up.unsqueeze(-1).unsqueeze(-1)
                lora_down = lora_down.unsqueeze(-1).unsqueeze(-1)
        
        # Convert to appropriate dtype
        lora_up = lora_up.to(diff.dtype)
        lora_down = lora_down.to(diff.dtype)
        alpha_tensor = torch.tensor(alpha, dtype=diff.dtype)
        
        return lora_up, lora_down, alpha_tensor
        
    except Exception as e:
        logger.error(f"SVD to LoRA conversion failed: {e}")
        return None, None, None


def merge_lora_weights_into_model(
    model: LBMModel,
    lora_weights: Dict[str, torch.Tensor],
    lora_scale: float = 1.0
) -> LBMModel:
    """
    Merge LoRA weights permanently into model weights (Path 2 approach).
    
    Args:
        model: LBM model to merge LoRA into
        lora_weights: LoRA weight tensors
        lora_scale: Scaling factor for LoRA
    
    Returns:
        Model with merged LoRA weights
    """
    logger.info(f"Merging LoRA weights into model with scale {lora_scale}")
    
    # Group LoRA weights
    lora_modules = {}
    for key, weight in lora_weights.items():
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
    
    # Apply LoRA deltas to model state dict
    model_state = model.state_dict()
    merged_count = 0
    
    for module_name, lora_data in lora_modules.items():
        if "up" not in lora_data or "down" not in lora_data:
            continue
        
        # Convert to state dict key
        state_key = "denoiser." + module_name.replace("lora_unet_", "")
        state_key = _convert_lora_name_to_state_key(state_key) + ".weight"
        
        if state_key not in model_state:
            continue
        
        # Compute LoRA delta
        lora_up = lora_data["up"]
        lora_down = lora_data["down"]
        alpha = lora_data.get("alpha", torch.tensor(16.0))
        
        # Simple delta computation for merging
        if len(lora_up.shape) == 2 and len(lora_down.shape) == 2:  # Linear
            delta = lora_up @ lora_down
        else:  # Conv - use simplified approach for merging
            # Flatten, multiply, reshape
            target_shape = model_state[state_key].shape
            lora_up_flat = lora_up.flatten(start_dim=1) if len(lora_up.shape) > 2 else lora_up
            lora_down_flat = lora_down.flatten(start_dim=1) if len(lora_down.shape) > 2 else lora_down
            
            try:
                if lora_up_flat.shape[0] == target_shape[0]:
                    delta = (lora_up_flat @ lora_down_flat.T).reshape(target_shape)
                else:
                    delta = (lora_down_flat @ lora_up_flat.T).reshape(target_shape)
            except:
                logger.warning(f"Could not merge {module_name}, skipping")
                continue
        
        # Apply with scaling
        rank = lora_up.shape[0] if len(lora_up.shape) >= 2 else 1
        alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        scale = (alpha_val / rank) * lora_scale
        
        model_state[state_key] = model_state[state_key] + delta.to(model_state[state_key].dtype) * scale
        merged_count += 1
    
    # Load merged state back
    model.load_state_dict(model_state)
    logger.info(f"Merged {merged_count} LoRA modules into model")
    
    return model


def _convert_lora_name_to_state_key(state_key: str) -> str:
    """Convert LoRA naming back to state dict key"""
    # Remove denoiser prefix for conversion
    path = state_key.replace("denoiser.", "")
    
    # Convert numbered patterns
    path = re.sub(r'_(\d+)_', r'.\1.', path)
    path = re.sub(r'_(\d+)$', r'.\1', path)
    
    # Fix LBM-specific patterns
    replacements = [
        ('mid_block_attentions', 'mid_block.attentions'),
        ('ff_net', 'ff.net'),
        ('attn1_to_', 'attn1.to_'),
    ]
    
    for old, new in replacements:
        path = path.replace(old, new)
    
    return "denoiser." + path


def validate_lora_compatibility(
    base_model: LBMModel,
    lora_weights: Dict[str, torch.Tensor],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate LoRA compatibility with base model following LBM conventions.
    
    Args:
        base_model: Base LBM model
        lora_weights: LoRA weight tensors
        strict: Whether to enforce strict compatibility
    
    Returns:
        Validation results
    """
    validation = {
        "compatible": True,
        "total_modules": 0,
        "compatible_modules": 0,
        "incompatible_modules": [],
        "warnings": []
    }
    
    # Group LoRA weights
    lora_modules = {}
    for key in lora_weights.keys():
        if ".lora_up.weight" in key:
            module_name = key.replace(".lora_up.weight", "")
            lora_modules[module_name] = key
    
    validation["total_modules"] = len(lora_modules)
    
    # Check each module
    for module_name in lora_modules:
        try:
            # Convert to model path
            model_path = module_name.replace("lora_unet_", "")
            model_path = re.sub(r'_(\d+)_', r'.\1.', model_path)
            model_path = re.sub(r'_(\d+)$', r'.\1', model_path)
            
            # Fix patterns
            replacements = [
                ('mid_block_attentions', 'mid_block.attentions'),
                ('ff_net', 'ff.net'),
                ('attn1_to_', 'attn1.to_'),
            ]
            
            for old, new in replacements:
                model_path = model_path.replace(old, new)
            
            # Check if module exists
            target_module = base_model.denoiser
            for part in model_path.split("."):
                target_module = getattr(target_module, part)
            
            if hasattr(target_module, 'weight'):
                validation["compatible_modules"] += 1
            else:
                validation["incompatible_modules"].append(f"{module_name}: no weight attribute")
                
        except AttributeError as e:
            validation["incompatible_modules"].append(f"{module_name}: {str(e)}")
    
    # Determine overall compatibility
    compatibility_ratio = validation["compatible_modules"] / validation["total_modules"]
    
    if strict and compatibility_ratio < 1.0:
        validation["compatible"] = False
    elif compatibility_ratio < 0.8:
        validation["compatible"] = False
        validation["warnings"].append(f"Low compatibility: {compatibility_ratio:.1%}")
    elif compatibility_ratio < 0.95:
        validation["warnings"].append(f"Some modules incompatible: {compatibility_ratio:.1%}")
    
    logger.info(f"LoRA compatibility: {validation['compatible_modules']}/{validation['total_modules']} modules compatible")
    
    return validation


def create_lora_training_config(
    base_lbm_config: Dict[str, Any],
    lora_config: LBMLoRAConfig,
    learning_rate: float = 1e-4,
    **training_kwargs
) -> Dict[str, Any]:
    """
    Create training configuration for LoRA fine-tuning following LBM patterns.
    
    Args:
        base_lbm_config: Base LBM configuration
        lora_config: LoRA configuration
        learning_rate: Learning rate for LoRA training
        **training_kwargs: Additional training parameters
    
    Returns:
        Training configuration dictionary
    """
    # Create LoRA-specific training config based on LBM patterns
    lora_training_config = base_lbm_config.copy()
    
    # Update for LoRA training
    lora_training_config.update({
        "learning_rate": learning_rate,  # Higher LR for LoRA
        "trainable_params": ["*.lora_*"],  # Only train LoRA parameters
        "optimizer_kwargs": {"weight_decay": 0.01},  # Light regularization
        "lora_config": lora_config.to_dict(),
        **training_kwargs
    })
    
    return lora_training_config


def _svd_to_lora(diff: torch.Tensor, rank: int, alpha: float) -> tuple:
    """Convert weight difference to LoRA via SVD"""
    try:
        # Handle different tensor types
        if len(diff.shape) == 4:  # Conv
            original_shape = diff.shape
            diff_2d = diff.flatten(start_dim=1)  # [out_ch, in_ch*kh*kw]
        else:  # Linear
            diff_2d = diff
            original_shape = None
        
        # SVD
        U, S, Vh = torch.linalg.svd(diff_2d.float())
        
        # Truncate to rank
        rank = min(rank, min(diff_2d.shape), len(S))
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]
        
        # Split singular values
        s_sqrt = torch.sqrt(torch.clamp(S_k, min=0.0))
        lora_up = U_k * s_sqrt.unsqueeze(0)
        lora_down = Vh_k * s_sqrt.unsqueeze(1)
        
        # Reshape for conv if needed
        if original_shape is not None:
            out_ch, in_ch, kh, kw = original_shape
            
            # Use LBM's LoRA format for conv layers
            if original_shape[2:] != (1, 1):  # 3x3 conv
                # Format: up=[rank, in_ch, kh, kw], down=[out_ch, rank, 1, 1]
                lora_up = lora_down.T.reshape(rank, in_ch, kh, kw)
                lora_down = (U_k * s_sqrt.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
            else:  # 1x1 conv
                lora_up = lora_up.unsqueeze(-1).unsqueeze(-1)
                lora_down = lora_down.unsqueeze(-1).unsqueeze(-1)
        
        # Create alpha tensor
        alpha_tensor = torch.tensor(alpha, dtype=diff.dtype)
        
        return lora_up.to(diff.dtype), lora_down.to(diff.dtype), alpha_tensor
        
    except Exception as e:
        logger.error(f"SVD to LoRA failed: {e}")
        return None, None, None
