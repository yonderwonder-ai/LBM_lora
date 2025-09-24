"""
LBM Model Utilities for LoRA Extraction and Application

This module provides utilities to:
1. Initialize LBM base models from SDXL
2. Handle LBM architectural modifications
3. Support LoRA extraction and application
"""

import os
import sys
import logging
import torch
from typing import List, Optional, Dict, Any
from diffusers import StableDiffusionXLPipeline, FlowMatchEulerDiscreteScheduler

# Add LBM source path
LBM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LBM_ROOT, "src"))

from lbm.models.lbm import LBMConfig, LBMModel  
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.models.embedders import ConditionerWrapper, LatentsConcatEmbedder, LatentsConcatEmbedderConfig
from lbm.trainer.utils import StateDictAdapter

logger = logging.getLogger(__name__)


def create_lbm_base_model(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    unet_input_channels: int = 4,
    vae_num_channels: int = 4,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    source_key: str = "source_image",
    target_key: str = "target_image", 
    mask_key: Optional[str] = None,
    save_base_path: Optional[str] = None,
    **kwargs
) -> LBMModel:
    """
    Creates LBM base model with architectural modifications from SDXL.
    This replicates the exact initialization process from train_lbm_surface.py
    
    Args:
        backbone_signature: SDXL model to use as base
        device: Device to load model on
        torch_dtype: Data type for model weights
        save_base_path: Path to save the base model weights
        **kwargs: Additional LBM configuration parameters
    
    Returns:
        LBMModel: Initialized LBM base model ready for fine-tuning
    """
    logger.info(f"Creating LBM base model from {backbone_signature}")
    
    if conditioning_images_keys is None:
        conditioning_images_keys = []
    if conditioning_masks_keys is None:
        conditioning_masks_keys = []
    
    # Step 1: Load SDXL pipeline 
    logger.info("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch_dtype,
    )
    
    # Step 2: Create LBM UNet with modified architecture
    logger.info("Creating LBM UNet architecture...")
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
        cross_attention_dim=[320, 640, 1280],  # LBM: Self-attention focused
        transformer_layers_per_block=[1, 2, 10],  # LBM: Specific layer distribution
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None, 
        attention_head_dim=[5, 10, 20],  # LBM: Specific attention heads
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,  # LBM: No add_embedding
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
    
    # Step 3: Extract and modify SDXL state dict
    logger.info("Adapting SDXL weights for LBM architecture...")
    state_dict = pipe.unet.state_dict()
    
    # Remove SDXL add_embedding layers (incompatible with LBM)
    layers_to_remove = [
        "add_embedding.linear_1.weight",
        "add_embedding.linear_1.bias", 
        "add_embedding.linear_2.weight",
        "add_embedding.linear_2.bias"
    ]
    
    for layer in layers_to_remove:
        if layer in state_dict:
            del state_dict[layer]
            logger.info(f"Removed incompatible layer: {layer}")
    
    # Step 4: Use StateDictAdapter to handle shape mismatches
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
        strategy="zeros",  # Initialize mismatched layers with zeros
    )
    
    # Step 5: Load adapted weights
    denoiser.load_state_dict(adapted_state_dict, strict=True)
    logger.info("Successfully loaded adapted SDXL weights into LBM UNet")
    
    # Clean up SDXL pipeline
    del pipe
    
    # Step 6: Setup conditioners (if needed)
    conditioners = []
    if conditioning_images_keys or conditioning_masks_keys:
        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)
    
    conditioner = ConditionerWrapper(conditioners=conditioners)
    
    # Step 7: Setup VAE
    logger.info("Setting up VAE...")
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch_dtype)
    
    # Step 8: Create LBM config
    config = LBMConfig(
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        **kwargs
    )
    
    # Step 9: Setup noise schedulers
    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler", 
    )
    
    # Step 10: Create final LBM model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch_dtype).to(device)
    
    # Step 11: Save base model if requested
    if save_base_path:
        logger.info(f"Saving LBM base model to {save_base_path}")
        # Create directory only if the path contains a directory
        dir_path = os.path.dirname(save_base_path)
        if dir_path:  # Only create directory if it's not empty
            os.makedirs(dir_path, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'backbone_signature': backbone_signature,
            'architecture': 'LBM',
            'torch_dtype': str(torch_dtype),
        }, save_base_path)
    
    logger.info("LBM base model creation completed successfully")
    return model


def get_lbm_target_modules_full() -> List[str]:
    """
    Returns complete list of target modules for full LBM LoRA coverage.
    
    Returns:
        List[str]: Module name patterns for LoRA targeting
    """
    return [
        # === ATTENTION LAYERS ===
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        
        # === FEED-FORWARD LAYERS ===
        "ff.net.0.proj", "ff.net.2",
        
        # === CONVOLUTION LAYERS ===
        "conv1", "conv2", "conv_shortcut",
        
        # === TIME EMBEDDING ===
        "time_emb_proj",
        
        # === INPUT/OUTPUT CONVOLUTIONS ===
        "conv_in", "conv_out",
        
        # === DOWNSAMPLING/UPSAMPLING ===
        "conv", "downsamplers.0.conv", "upsamplers.0.conv",
    ]


def get_lbm_target_modules_attention_only() -> List[str]:
    """
    Returns attention-only target modules for minimal LBM LoRA.
    
    Returns:
        List[str]: Attention module patterns only
    """
    return [
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    ]


def get_lbm_target_modules_extended() -> List[str]:
    """
    Returns extended target modules (attention + feed-forward).
    
    Returns:
        List[str]: Attention and FF module patterns
    """
    return [
        # Attention
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        # Feed-forward
        "ff.net.0.proj", "ff.net.2",
    ]


def load_lbm_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16
) -> LBMModel:
    """
    Loads LBM model from checkpoint file (.safetensors or .ckpt/.pth).
    
    Args:
        checkpoint_path: Path to LBM checkpoint
        device: Device to load model on
        torch_dtype: Data type for model weights
    
    Returns:
        LBMModel: Loaded LBM model
    """
    logger.info(f"Loading LBM model from {checkpoint_path}")
    
    # Handle different file formats
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
        
        # For safetensors files, we need to reconstruct config from filename/metadata
        # This is a common case for HuggingFace models
        # Create a default LBM config for now
        config_dict = {
            'source_key': 'source_image',
            'target_key': 'target_image', 
            'latent_loss_weight': 1.0,
            'latent_loss_type': 'l2',
            'pixel_loss_type': 'lpips',
            'pixel_loss_weight': 0.0,
            'timestep_sampling': 'uniform',
            'bridge_noise_sigma': 0.001
        }
        backbone_signature = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Create base model
        model = create_lbm_base_model(
            backbone_signature=backbone_signature,
            device=device,
            torch_dtype=torch_dtype,
            **config_dict
        )
        
        # Load weights directly (safetensors doesn't have the wrapper structure)
        model.load_state_dict(checkpoint, strict=False)
        
    else:
        # Handle .ckpt/.pth files with torch.load
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Check if it's a direct state dict or wrapped
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            config_dict = checkpoint.get('config', {})
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint format
            model_state = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            if any(k.startswith('model.') for k in model_state.keys()):
                model_state = {k[6:] if k.startswith('model.') else k: v for k, v in model_state.items()}
            config_dict = {}
        else:
            # Direct state dict
            model_state = checkpoint
            config_dict = {}
        
        # Fill in default config if missing
        if not config_dict:
            config_dict = {
                'source_key': 'source_image',
                'target_key': 'target_image',
                'latent_loss_weight': 1.0,
                'latent_loss_type': 'l2', 
                'pixel_loss_type': 'lpips',
                'pixel_loss_weight': 0.0,
                'timestep_sampling': 'uniform',
                'bridge_noise_sigma': 0.001
            }
        
        backbone_signature = checkpoint.get('backbone_signature', "stabilityai/stable-diffusion-xl-base-1.0")
        
        # Create base model
        model = create_lbm_base_model(
            backbone_signature=backbone_signature,
            device=device, 
            torch_dtype=torch_dtype,
            **config_dict
        )
        
        # Load weights
        model.load_state_dict(model_state, strict=False)
    
    logger.info("LBM model loaded successfully")
    return model


def save_lbm_config(config: Dict[str, Any], save_path: str):
    """
    Saves LBM configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    import yaml
    
    # Create directory only if the path contains a directory
    dir_path = os.path.dirname(save_path)
    if dir_path:  # Only create directory if it's not empty
        os.makedirs(dir_path, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"LBM config saved to {save_path}")
