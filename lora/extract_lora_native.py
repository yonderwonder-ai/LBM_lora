"""
LBM LoRA Extraction with Native Naming

This version uses LBM's actual state dict keys instead of converting names.
This should solve the mid-block naming issues and achieve >95% success rate.
"""

import sys
import os
import argparse
import json
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import logging
from typing import List, Optional, Dict, Any

# Add LBM utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lbm_utils import (
    create_lbm_base_model,
    load_lbm_model_from_checkpoint,
    get_lbm_target_modules_full,
    get_lbm_target_modules_attention_only,
    get_lbm_target_modules_extended,
    save_lbm_config
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_target_modules_native(
    model: torch.nn.Module, 
    target_patterns: List[str]
) -> Dict[str, torch.nn.Module]:
    """
    Find target modules using LBM's native naming (state dict keys).
    
    Args:
        model: LBM model (denoiser)
        target_patterns: List of patterns to match (e.g., "attn1.to_q")
    
    Returns:
        Dict mapping state dict keys to modules
    """
    target_modules = {}
    
    logger.info(f"Searching for patterns: {target_patterns}")
    
    # Use named_modules to get actual LBM naming
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            # Check if this module matches any target pattern
            for pattern in target_patterns:
                if pattern in name and name.endswith(pattern):
                    # Use the actual state dict key format
                    state_key = f"denoiser.{name}"
                    target_modules[state_key] = module
                    logger.debug(f"Found target: {state_key}")
                    break
    
    logger.info(f"Found {len(target_modules)} target modules using native naming")
    return target_modules


def calculate_lora_delta_native(
    base_weight: torch.Tensor,
    tuned_weight: torch.Tensor,
    rank: int,
    alpha: float,
    device: str,
    save_dtype: torch.dtype
) -> tuple:
    """
    Calculate LoRA components using SVD with correct tensor format.
    """
    # Calculate difference
    diff = tuned_weight - base_weight
    
    # Check if difference is significant
    max_diff = torch.max(torch.abs(diff)).item()
    if max_diff < 0.001:
        return None, None, None
    
    # Prepare for SVD
    original_shape = diff.shape
    if len(diff.shape) == 4:  # Conv2d
        diff_2d = diff.flatten(start_dim=1)  # [out_ch, in_ch*kh*kw]
    else:  # Linear
        diff_2d = diff
    
    # Perform SVD
    try:
        U, S, Vh = torch.linalg.svd(diff_2d.float())
    except Exception as e:
        logger.error(f"SVD failed: {e}")
        return None, None, None
    
    # Determine effective rank
    rank = min(rank, min(diff_2d.shape), len(S))
    
    # Create LoRA components
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    
    # Split singular values evenly
    s_sqrt = torch.sqrt(torch.clamp(S_k, min=0.0))
    lora_up = U_k * s_sqrt.unsqueeze(0)      # [out_dim, rank]
    lora_down = Vh_k * s_sqrt.unsqueeze(1)   # [rank, in_dim]
    
    # Reshape for conv layers if needed
    if len(original_shape) == 4:  # Conv layer
        out_ch, in_ch, kh, kw = original_shape
        
        # LBM format for conv layers
        if original_shape[2:] == (1, 1):  # 1x1 conv
            lora_up = lora_up.unsqueeze(-1).unsqueeze(-1)  # [out_ch, rank, 1, 1]
            lora_down = lora_down.unsqueeze(-1).unsqueeze(-1)  # [rank, in_ch, 1, 1]
        else:  # 3x3 conv
            # More complex reshaping for 3x3 conv
            lora_up = lora_up.unsqueeze(-1).unsqueeze(-1)  # [out_ch, rank, 1, 1]
            lora_down = lora_down.reshape(rank, in_ch, kh, kw)  # [rank, in_ch, kh, kw]
    
    # Convert to target dtype and ensure contiguous
    lora_up = lora_up.to(device=device, dtype=save_dtype).contiguous()
    lora_down = lora_down.to(device=device, dtype=save_dtype).contiguous()
    alpha_tensor = torch.tensor(alpha, dtype=save_dtype, device=device)
    
    return lora_up, lora_down, alpha_tensor


def extract_lora_native(
    base_model_path: str = None,
    tuned_model_path: str = None,
    save_to: str = None,
    target_coverage: str = "full",
    rank: int = 32,
    conv_rank: int = 16,
    device: str = None,
    save_precision: str = "bf16",
    min_diff: float = 0.001,
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    create_base_model: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    Extract LoRA using LBM's native naming scheme.
    """
    # Setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype_mapping = {
        "float": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    save_dtype = dtype_mapping.get(save_precision, torch.float32)
    
    logger.info("Starting LBM LoRA extraction with native naming...")
    logger.info(f"Device: {device}, Target coverage: {target_coverage}")
    logger.info(f"Ranks: Linear={rank}, Conv={conv_rank}")
    
    logger.info("🔄 Processing ALL denoiser layers (no pattern filtering)")
    
    # Load or create base model
    if create_base_model or base_model_path is None:
        logger.info("Creating LBM base model from SDXL...")
        base_save_path = save_to.replace(".safetensors", "_base.safetensors") if save_to else None
        base_model = create_lbm_base_model(
            backbone_signature=backbone_signature,
            device=device,
            save_base_path=base_save_path,
            **kwargs
        )
    else:
        logger.info(f"Loading LBM base model from {base_model_path}")
        base_model = load_lbm_model_from_checkpoint(base_model_path, device=device)
    
    # Load tuned model
    logger.info(f"Loading fine-tuned model from {tuned_model_path}")
    tuned_model = load_lbm_model_from_checkpoint(tuned_model_path, device=device)
    
    # Get ALL denoiser layers (no filtering by patterns)
    base_state_dict = base_model.denoiser.state_dict()
    tuned_state_dict = tuned_model.denoiser.state_dict()
    
    # Find layers that exist in both models and have weight tensors
    all_denoiser_layers = {}
    for key, tensor in base_state_dict.items():
        if key in tuned_state_dict and key.endswith('.weight') and tensor.dim() >= 2:
            all_denoiser_layers[key] = tensor
    
    logger.info(f"📊 Found {len(all_denoiser_layers)} denoiser weight layers to analyze")
    
    # Extract LoRA weights
    logger.info("Extracting LoRA weights using native keys...")
    lora_weights = {}
    processed_count = 0
    extracted_count = 0
    
    with torch.no_grad():
        for state_key in tqdm(all_denoiser_layers.keys(), desc="Processing ALL denoiser layers"):
            processed_count += 1
            
            # Get weights directly from state dict
            base_weight = base_state_dict[state_key].to(device)
            tuned_weight = tuned_state_dict[state_key].to(device)
            
            # Check if layer has significant changes
            delta = tuned_weight - base_weight
            max_change = torch.max(torch.abs(delta)).item()
            
            # Skip layers with insignificant changes (likely just numerical noise)
            if max_change < min_diff:
                if verbose:
                    logger.debug(f"✗ Skipped {state_key} (max change: {max_change:.6f} < {min_diff})")
                continue
            
            # Determine rank based on layer type
            is_conv = len(base_weight.shape) == 4
            effective_rank = conv_rank if is_conv else rank
            effective_alpha = conv_rank if is_conv else rank
            
            # Extract LoRA
            lora_up, lora_down, alpha_tensor = calculate_lora_delta_native(
                base_weight,
                tuned_weight,
                effective_rank,
                effective_alpha,
                device,
                save_dtype
            )
            
            if lora_up is not None:
                # Use denoiser.layer_name format for LoRA keys
                lora_key_base = f"denoiser.{state_key.replace('.weight', '')}"
                
                lora_weights[f"{lora_key_base}.lora_up.weight"] = lora_up
                lora_weights[f"{lora_key_base}.lora_down.weight"] = lora_down
                lora_weights[f"{lora_key_base}.alpha"] = alpha_tensor
                
                extracted_count += 1
                
                if verbose:
                    layer_type = "CONV" if is_conv else "LINEAR"
                    logger.info(f"✓ {lora_key_base:80} | {layer_type} | rank: {effective_rank:2d} | diff: {max_change:.6f}")
            else:
                if verbose:
                    logger.debug(f"✗ Failed LoRA extraction for {state_key} (SVD issue)")
    
    logger.info(f"📊 EXTRACTION SUMMARY:")
    logger.info(f"   Total denoiser layers analyzed: {processed_count}")
    logger.info(f"   Layers with significant changes: {extracted_count}")
    logger.info(f"   Coverage: {extracted_count/processed_count*100:.1f}% of analyzed layers")
    logger.info(f"   Change threshold: {min_diff}")
    
    # Prepare metadata
    metadata = {
        "ss_base_model_version": "sdxl_v10",
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(rank),
        "ss_network_args": json.dumps({
            "conv_dim": str(conv_rank),
            "conv_alpha": str(conv_rank)
        }),
        "lbm_extraction_method": "all_layers_analysis",
        "lbm_total_analyzed": str(processed_count),
        "lbm_total_extracted": str(extracted_count),
        "lbm_backbone_signature": backbone_signature,
        "lbm_architecture": "true",
        "lbm_native_naming": "true",  # Flag to indicate native naming
        "ss_creation_time": str(int(time.time())),
    }
    
    # Save LoRA
    if save_to:
        dir_path = os.path.dirname(save_to)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        save_file(lora_weights, save_to, metadata=metadata)
        logger.info(f"LoRA weights saved to: {save_to}")
        
        # Save config
        config_path = save_to.replace(".safetensors", "_config.yaml")
        config = {
            "base_model": backbone_signature,
            "extraction_method": "all_layers_analysis",
            "rank": rank,
            "conv_rank": conv_rank,
            "architecture": "LBM",
            "native_naming": True,
            "total_analyzed_layers": processed_count,
            "total_lora_modules": extracted_count,
            "change_threshold": min_diff,
            "coverage_percentage": f"{extracted_count/processed_count*100:.1f}%",
        }
        save_lbm_config(config, config_path)
    
    # Cleanup
    del base_model, tuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Native LoRA extraction completed. Extracted {extracted_count} modules.")
    return lora_weights


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA using LBM native naming")
    
    # Model paths
    parser.add_argument("--base_model_path", type=str,
                       help="Path to base LBM model")
    parser.add_argument("--tuned_model_path", type=str, required=True,
                       help="Path to fine-tuned LBM model")
    parser.add_argument("--save_to", type=str, required=True,
                       help="Output path for LoRA weights")
    
    # LoRA configuration
    parser.add_argument("--target_coverage", type=str, default="full",
                       choices=["full", "attention", "extended"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--conv_rank", type=int, default=16)
    
    # Technical settings
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_precision", type=str, default="bf16",
                       choices=["float", "fp16", "bf16"])
    parser.add_argument("--min_diff", type=float, default=0.001)
    
    # Base model creation
    parser.add_argument("--create_base_model", action="store_true")
    parser.add_argument("--backbone_signature", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    
    # Other
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Run extraction
    extract_lora_native(**vars(args))


if __name__ == "__main__":
    main()
