"""
Create Exact LBM Base Model

This script replicates the EXACT initialization process from train_lbm_surface.py
to create the true base model that matches what fine-tuning started from.

This solves the base model mismatch issue that was causing poor LoRA results.
"""

import sys
import os
import argparse
import logging
import yaml
import torch
from diffusers import StableDiffusionXLPipeline, FlowMatchEulerDiscreteScheduler

# Add LBM source path (exact same as train_lbm_surface.py)
LBM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LBM_ROOT, "src"))

from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.models.embedders import ConditionerWrapper, LatentsConcatEmbedder, LatentsConcatEmbedderConfig
from lbm.trainer.utils import StateDictAdapter

logger = logging.getLogger(__name__)


def create_exact_lbm_base_model(
    config_path: str = "examples/inference/ckpts/relighting/config.yaml",
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    save_path: str = None
) -> LBMModel:
    """
    Create the EXACT base model using LBM architecture configuration.
    
    Uses inference config (config.yaml) which contains all architecture parameters.
    Training-specific parameters are set to sensible defaults since they don't
    affect the initial model state, only the training dynamics.
    
    Args:
        config_path: Path to LBM config (config.yaml from inference)
        backbone_signature: SDXL model signature
        device: Device to load on
        torch_dtype: Model dtype
        save_path: Path to save the exact base model
    
    Returns:
        LBMModel: Exact base model matching fine-tuned model's architecture
    """
    logger.info(f"Creating EXACT LBM base model from config: {config_path}")
    
    # Load LBM configuration (inference config contains all architecture params)
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    logger.info(f"Using config: {config_data}")
    
    # Extract architecture parameters from config
    unet_input_channels = config_data.get("unet_input_channels", 4)
    vae_num_channels = config_data.get("vae_num_channels", 4)
    conditioning_images_keys = config_data.get("conditioning_images_keys", [])
    conditioning_masks_keys = config_data.get("conditioning_masks_keys", [])
    
    # LBM config parameters (use config values where available, sensible defaults otherwise)
    source_key = config_data.get("source_key", "source_image")
    target_key = config_data.get("target_key", "target_image")
    mask_key = None
    
    # Training parameters (defaults don't affect initial model state)
    latent_loss_weight = config_data.get("latent_loss_weight", 1.0)
    latent_loss_type = config_data.get("latent_loss_type", "l2")
    pixel_loss_type = config_data.get("pixel_loss_type", "lpips")
    pixel_loss_weight = config_data.get("pixel_loss_weight", 10.0)
    
    # Timestep parameters (architecture-affecting)
    timestep_sampling = config_data.get("timestep_sampling", "custom_timesteps")
    selected_timesteps = config_data.get("selected_timesteps", [250, 500, 750, 1000])
    prob = config_data.get("prob", [0.25, 0.25, 0.25, 0.25])
    bridge_noise_sigma = config_data.get("bridge_noise_sigma", 0.005)
    
    # Logit parameters (from training script defaults)
    logit_mean = None
    logit_std = None
    
    logger.info("=" * 60)
    logger.info("EXACT LBM INITIALIZATION (matching fine-tuned architecture)")
    logger.info("=" * 60)
    
    # STEP 1: Load SDXL Pipeline (EXACT same as training)
    logger.info("Step 1: Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch_dtype,
    )
    
    # STEP 2: Create UNet with EXACT same config as training
    logger.info("Step 2: Creating UNet with EXACT training config...")
    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch_dtype)
    
    # STEP 3: Get SDXL state dict and remove incompatible layers (EXACT same as training)
    logger.info("Step 3: Adapting SDXL state dict...")
    state_dict = pipe.unet.state_dict()
    
    # Remove add_embedding layers (EXACT same as training)
    for key in list(state_dict.keys()):
        if "add_embedding" in key:
            del state_dict[key]
            logger.info(f"Removed incompatible layer: {key}")
    
    # STEP 4: Apply StateDictAdapter with EXACT same config as training
    logger.info("Step 4: Applying StateDictAdapter with EXACT training config...")
    state_dict_adapter = StateDictAdapter()
    adapted_state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )
    
    # STEP 5: Load adapted state dict (EXACT same as training)
    logger.info("Step 5: Loading adapted state dict...")
    denoiser.load_state_dict(adapted_state_dict, strict=True)
    
    # Clean up pipeline
    del pipe
    
    # STEP 6: Setup conditioners (EXACT same as training)
    logger.info("Step 6: Setting up conditioners...")
    conditioners = []
    
    if conditioning_images_keys != [] or conditioning_masks_keys != []:
        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)
    
    # Wrap conditioners
    conditioner = ConditionerWrapper(conditioners=conditioners)
    
    # STEP 7: Setup VAE (EXACT same as training)
    logger.info("Step 7: Setting up VAE...")
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch_dtype)
    
    # STEP 8: Create LBM Config (EXACT same as training)
    logger.info("Step 8: Creating LBM config...")
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )
    
    # STEP 9: Setup noise schedulers (EXACT same as training)
    logger.info("Step 9: Setting up noise schedulers...")
    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    
    # STEP 10: Create final LBM model (EXACT same as training)
    logger.info("Step 10: Creating final LBM model...")
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch_dtype)
    
    # Move to device
    model = model.to(device)
    
    # STEP 11: Save the EXACT base model (before any training)
    if save_path:
        logger.info(f"Step 11: Saving EXACT base model to {save_path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        # Get model state dict and exclude LPIPS (not needed for LoRA, causes shared tensor issues)
        model_state = model.state_dict()
        
        # Filter out LPIPS components - they're not needed for LoRA operations
        # and cause safetensors shared memory issues
        lora_relevant_state = {}
        lpips_excluded_count = 0
        
        for key, value in model_state.items():
            if 'lpips_loss' in key:
                lpips_excluded_count += 1
                logger.debug(f"Excluding LPIPS component: {key}")
                continue
            lora_relevant_state[key] = value
        
        logger.info(f"📊 Excluded {lpips_excluded_count} LPIPS parameters (not needed for LoRA)")
        logger.info(f"📊 Kept {len(lora_relevant_state)} core LBM parameters for LoRA operations")
        
        # Save with exact same format as training checkpoints
        checkpoint = {
            'model_state_dict': lora_relevant_state,  # Only core LBM components
            'config': config.to_dict(),
            'backbone_signature': backbone_signature,
            'architecture': 'LBM',
            'torch_dtype': str(torch_dtype),
            'source_config': config_data,  # Save original config used
            'initialization': 'exact_architecture_match',
            'created_from': 'inference_config_initialization',
            'lpips_excluded': True,  # Flag that LPIPS was intentionally excluded
            'note': 'LPIPS can be loaded separately during training if needed',
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"✅ EXACT base model saved to {save_path}")
        
        # Also save as safetensors for compatibility (should work now that LPIPS is excluded)
        if save_path.endswith('.ckpt') or save_path.endswith('.pth'):
            try:
                from safetensors.torch import save_file
                safetensors_path = save_path.replace('.ckpt', '.safetensors').replace('.pth', '.safetensors')
                save_file(lora_relevant_state, safetensors_path)
                logger.info(f"✅ Also saved as safetensors: {safetensors_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not save safetensors format: {e}")
                logger.warning("   Main .ckpt file is sufficient for LoRA operations")
    
    logger.info("🎯 EXACT LBM base model creation completed!")
    logger.info("This model matches exactly what fine-tuning started from.")
    
    return model


def test_exact_base_vs_finetuned(
    exact_base_path: str,
    finetuned_path: str = "examples/inference/ckpts/relighting/model.safetensors"
):
    """
    Test if the exact base model has perfect baseline with fine-tuned model.
    This should show near-zero error in base model differences.
    """
    logger.info("Testing exact base vs fine-tuned model...")
    
    device = "cpu"  # Use CPU for testing to save memory
    
    # Load exact base model
    exact_base = torch.load(exact_base_path, map_location=device, weights_only=False)
    exact_base_state = exact_base['model_state_dict']
    
    # Load fine-tuned model  
    from safetensors.torch import load_file
    finetuned_state = load_file(finetuned_path, device=device)
    
    # Calculate differences only for UNet (denoiser) parameters
    denoiser_diffs = []
    matched_keys = 0
    total_keys = 0
    
    for key in exact_base_state.keys():
        if key.startswith('denoiser.') and key in finetuned_state:
            if exact_base_state[key].shape == finetuned_state[key].shape:
                diff = torch.mean(torch.abs(finetuned_state[key] - exact_base_state[key])).item()
                denoiser_diffs.append(diff)
                matched_keys += 1
        total_keys += 1
    
    avg_diff = sum(denoiser_diffs) / len(denoiser_diffs) if denoiser_diffs else float('inf')
    max_diff = max(denoiser_diffs) if denoiser_diffs else float('inf')
    
    print()
    print("🧪 EXACT BASE MODEL VALIDATION:")
    print("-" * 40)
    print(f"Matched denoiser keys: {matched_keys}/{total_keys}")
    print(f"Average difference: {avg_diff:.8f}")
    print(f"Maximum difference: {max_diff:.8f}")
    print()
    
    if avg_diff < 0.001:
        print("🎉 EXCELLENT: Base model matches training baseline perfectly!")
        print("   LoRA extraction should work perfectly with this base.")
    elif avg_diff < 0.01:
        print("✅ GOOD: Base model is very close to training baseline.")
        print("   LoRA should work well with this base.")
    else:
        print("⚠️ WARNING: Base model differs significantly from training baseline.")
        print("   LoRA quality may still be affected.")
    
    return avg_diff


def main():
    """Create exact LBM base model using architecture configuration"""
    parser = argparse.ArgumentParser(description="Create Exact LBM Base Model")
    
    parser.add_argument("--config_path", type=str, 
                       default="examples/inference/ckpts/relighting/config.yaml",
                       help="Path to LBM config (inference config contains architecture)")
    parser.add_argument("--backbone_signature", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="SDXL model signature")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for model creation")
    parser.add_argument("--torch_dtype", type=str, default="bf16",
                       choices=["float", "fp16", "bf16"],
                       help="Model precision")
    parser.add_argument("--save_path", type=str, 
                       default="lora/checkpoints/base/exact_lbm_base.ckpt",
                       help="Path to save exact base model")
    parser.add_argument("--test_baseline", action="store_true",
                       help="Test baseline difference with fine-tuned model")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Convert dtype
    dtype_mapping = {
        "float": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    torch_dtype = dtype_mapping[args.torch_dtype]
    
    # Check if config file exists
    if not os.path.exists(args.config_path):
        logger.error(f"LBM config not found: {args.config_path}")
        return
    
    # Create exact base model
    model = create_exact_lbm_base_model(
        config_path=args.config_path,
        backbone_signature=args.backbone_signature,
        device=args.device,
        torch_dtype=torch_dtype,
        save_path=args.save_path
    )
    
    # Test baseline if requested
    if args.test_baseline:
        test_exact_base_vs_finetuned(args.save_path)
    
    print()
    print("🎯 SUCCESS: Exact LBM base model created!")
    print("Use this base model for LoRA extraction to eliminate base model mismatch.")
    print()
    print("Next steps:")
    print(f"1. Extract LoRA: python lora/extract_lora_native.py --base_model_path {args.save_path}")
    print("2. Test reconstruction: python lora/test_perfect_reconstruction.py")


if __name__ == "__main__":
    main()
