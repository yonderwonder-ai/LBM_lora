"""
LBM LoRA Inference with Native Naming

This version uses LBM's actual state dict keys directly.
No name conversion needed - should achieve >95% success rate.
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any
from PIL import Image

import torch
from safetensors.torch import load_file

# Add LBM paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lbm_utils import create_lbm_base_model, load_lbm_model_from_checkpoint
from lbm.inference import evaluate

logger = logging.getLogger(__name__)


class NativeLBMLoRAInference:
    """
    LBM LoRA inference using native state dict keys.
    No name conversion - direct state dict manipulation.
    """
    
    def __init__(
        self,
        base_model_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_scale: float = 1.0,
        attention_only: bool = False,
        backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
        verbose: bool = False
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_scale = lora_scale
        self.attention_only = attention_only
        self.backbone_signature = backbone_signature
        self.verbose = verbose
        
        # Load base model
        self.model = self._load_base_model(base_model_path)
        
        # Apply LoRA using native state dict manipulation
        if lora_path:
            self.lora_weights = self._load_lora_weights(lora_path)
            self._apply_lora_native()
        else:
            self.lora_weights = None
            
        logger.info("Native LBM LoRA inference model initialized")
    
    def _load_base_model(self, base_model_path: Optional[str]):
        """Load base LBM model"""
        if base_model_path and os.path.exists(base_model_path):
            logger.info(f"Loading base model from {base_model_path}")
            return load_lbm_model_from_checkpoint(
                base_model_path,
                device=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            logger.info(f"Creating base model from {self.backbone_signature}")
            return create_lbm_base_model(
                backbone_signature=self.backbone_signature,
                device=self.device,
                torch_dtype=self.torch_dtype
            )
    
    def _load_lora_weights(self, lora_path: str) -> Dict[str, torch.Tensor]:
        """Load LoRA weights with native keys"""
        logger.info(f"Loading LoRA weights from {lora_path}")
        lora_state_dict = load_file(lora_path)
        
        # Move to device and dtype
        for key in lora_state_dict:
            lora_state_dict[key] = lora_state_dict[key].to(
                device=self.device, dtype=self.torch_dtype
            )
        
        # Filter to attention layers if specified
        if self.attention_only:
            filtered_dict = {}
            for key, value in lora_state_dict.items():
                if any(attn_key in key for attn_key in ["attn1.to", "attn2.to"]):  # Both self and cross attention
                    filtered_dict[key] = value
            
            attention_modules = len([k for k in filtered_dict.keys() if k.endswith('.lora_up.weight')])
            logger.info(f"Filtered to {attention_modules} attention modules (native naming)")
            return filtered_dict
        
        total_modules = len([k for k in lora_state_dict.keys() if k.endswith('.lora_up.weight')])
        logger.info(f"Loaded {total_modules} LoRA modules (native naming)")
        return lora_state_dict
    
    def _apply_lora_native(self):
        """Apply LoRA weights using native state dict manipulation"""
        if not self.lora_weights:
            return
        
        logger.info("Applying LoRA using native state dict keys...")
        
        # Get current model state dict
        model_state = self.model.state_dict()
        
        # Group LoRA weights by module
        lora_modules = {}
        for key, weight in self.lora_weights.items():
            if ".lora_up.weight" in key or ".lora_down.weight" in key or ".alpha" in key:
                # Extract base key (everything before .lora_up/.lora_down/.alpha)
                if ".lora_up.weight" in key:
                    base_key = key.replace(".lora_up.weight", "")
                elif ".lora_down.weight" in key:
                    base_key = key.replace(".lora_down.weight", "")
                elif ".alpha" in key:
                    base_key = key.replace(".alpha", "")
                
                if base_key not in lora_modules:
                    lora_modules[base_key] = {}
                
                if ".lora_up.weight" in key:
                    lora_modules[base_key]["up"] = weight
                elif ".lora_down.weight" in key:
                    lora_modules[base_key]["down"] = weight
                elif ".alpha" in key:
                    lora_modules[base_key]["alpha"] = weight
        
        applied_count = 0
        failed_count = 0
        
        # Apply LoRA deltas directly to state dict
        for base_key, lora_data in lora_modules.items():
            if "up" not in lora_data or "down" not in lora_data:
                continue
            
            # The target state dict key
            target_key = f"{base_key}.weight"
            
            if target_key not in model_state:
                if self.verbose:
                    logger.debug(f"✗ Target key not found: {target_key}")
                failed_count += 1
                continue
            
            try:
                # Apply LoRA delta using native keys
                success = self._apply_lora_delta_native(
                    model_state, target_key, lora_data
                )
                
                if success:
                    applied_count += 1
                    if self.verbose:
                        logger.debug(f"✓ Applied LoRA: {base_key}")
                else:
                    failed_count += 1
                    if self.verbose:
                        logger.debug(f"✗ Failed LoRA: {base_key}")
                        
            except Exception as e:
                failed_count += 1
                if self.verbose:
                    logger.debug(f"✗ Exception in {base_key}: {e}")
        
        # Load updated state dict back to model
        self.model.load_state_dict(model_state)
        
        success_rate = applied_count / (applied_count + failed_count) * 100 if (applied_count + failed_count) > 0 else 0
        layer_type = "attention" if self.attention_only else "all"
        logger.info(f"Native LoRA applied ({layer_type} layers): {applied_count} success, {failed_count} failed ({success_rate:.1f}% success rate)")
    
    def _apply_lora_delta_native(
        self, 
        model_state: Dict[str, torch.Tensor], 
        target_key: str, 
        lora_data: Dict[str, torch.Tensor]
    ) -> bool:
        """Apply LoRA delta directly to state dict"""
        try:
            lora_up = lora_data["up"]
            lora_down = lora_data["down"]
            alpha = lora_data.get("alpha", torch.tensor(16.0))
            
            # Get target weight
            target_weight = model_state[target_key]
            
            # Handle both Linear (2D) and Conv (4D) layers
            if len(lora_up.shape) == 2 and len(lora_down.shape) == 2:
                # Linear layers: [out_features, rank] @ [rank, in_features]
                if lora_up.shape[1] == lora_down.shape[0]:  # Standard order
                    delta = lora_up @ lora_down
                    rank = lora_up.shape[1]
                elif lora_down.shape[1] == lora_up.shape[0]:  # Swapped order
                    delta = lora_down @ lora_up
                    rank = lora_up.shape[0]
                else:
                    if self.verbose:
                        logger.debug(f"Linear shape mismatch: up={lora_up.shape}, down={lora_down.shape}")
                    return False
                    
            elif len(lora_up.shape) == 4 or len(lora_down.shape) == 4:
                # Conv layers: reshape to 2D, multiply, reshape back
                try:
                    # Reshape conv tensors for matrix multiplication
                    if len(lora_up.shape) == 4:  # Conv -> Linear
                        up_2d = lora_up.flatten(1)  # [out_ch, rank*kernel_size]
                    else:
                        up_2d = lora_up  # Already 2D

                    if len(lora_down.shape) == 4:  # Conv -> Linear  
                        down_2d = lora_down.flatten(1)  # [rank, in_ch*kernel_size]
                    else:
                        down_2d = lora_down  # Already 2D

                    # Matrix multiplication
                    if up_2d.shape[1] == down_2d.shape[0]:
                        delta_2d = up_2d @ down_2d
                        rank = up_2d.shape[1]
                    elif up_2d.shape[0] == down_2d.shape[1]:
                        delta_2d = down_2d @ up_2d  # Swapped
                        rank = up_2d.shape[0]
                    else:
                        if self.verbose:
                            logger.debug(f"Conv shape mismatch: up_2d={up_2d.shape}, down_2d={down_2d.shape}")
                        return False

                    # Reshape back to original weight shape
                    delta = delta_2d.reshape(target_weight.shape)
                    
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Conv reshaping failed: {e}")
                    return False
                    
            else:
                if self.verbose:
                    logger.debug(f"Unsupported tensor dimensions: up={lora_up.shape}, down={lora_down.shape}")
                return False
            
            # Check target shape match
            if delta.shape != target_weight.shape:
                if self.verbose:
                    logger.debug(f"Delta shape mismatch: {delta.shape} vs {target_weight.shape}")
                return False
            
            # Apply with scaling
            alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            scale = (alpha_val / rank) * self.lora_scale
            
            # Update state dict
            model_state[target_key] = target_weight + delta.to(target_weight.dtype) * scale
            
            return True
            
        except Exception as e:
            if self.verbose:
                logger.debug(f"LoRA delta application failed: {e}")
            return False
    
    def generate(
        self, 
        source_image: Image.Image,
        num_inference_steps: int = 1,
        **kwargs
    ) -> Image.Image:
        """Generate output using LBM with native LoRA"""
        return evaluate(self.model, source_image, num_inference_steps)


def main():
    """Main function with debug logging"""
    parser = argparse.ArgumentParser(description="LBM LoRA Inference with Native Naming")
    
    # Model configuration
    parser.add_argument("--base_model_path", type=str,
                       help="Path to base LBM model")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to LoRA weights")
    
    # Inference settings
    parser.add_argument("--source_image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save output image")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    
    # Technical settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bf16",
                       choices=["float", "fp16", "bf16"])
    parser.add_argument("--backbone_signature", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    
    # Layer selection  
    parser.add_argument("--attention_only", action="store_true", default=False,
                       help="Apply LoRA only to attention layers (default: apply to all layers)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true",
                       help="Enable detailed debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=log_level,
                       format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Convert dtype
    dtype_mapping = {
        "float": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    torch_dtype = dtype_mapping[args.torch_dtype]
    
    # Initialize model
    logger.info("Initializing native LBM LoRA inference...")
    native_lora = NativeLBMLoRAInference(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        device=args.device,
        torch_dtype=torch_dtype,
        lora_scale=args.lora_scale,
        attention_only=args.attention_only,
        backbone_signature=args.backbone_signature,
        verbose=args.verbose or args.debug
    )
    
    # Load and process image
    logger.info(f"Loading source image: {args.source_image}")
    source_image = Image.open(args.source_image).convert("RGB")
    
    # Generate output
    logger.info("Generating output with native LoRA...")
    with torch.inference_mode():
        output_image = native_lora.generate(
            source_image=source_image,
            num_inference_steps=args.num_inference_steps
        )
    
    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_image.save(args.output_path)
    source_image.save(args.output_path.replace(".jpg", "_input.jpg"))
    
    logger.info(f"Results saved to {args.output_path}")
    logger.info("Native LoRA inference completed successfully")


if __name__ == "__main__":
    main()
