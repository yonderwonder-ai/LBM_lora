from typing import List, Literal, Optional
from pydantic.dataclasses import dataclass
from ..base import ModelConfig


@dataclass
class LBMLoRAConfig(ModelConfig):
    """This is the Config for LBM LoRA Model class which defines all the useful parameters for LoRA adaptation.

    Args:

        target_modules (List[str]):
            List of module name patterns to target for LoRA adaptation.
            Defaults to attention layers only.

        rank (int):
            Rank for linear layer LoRA adaptations. Higher rank = more adaptation capacity.
            Defaults to 16.

        conv_rank (int):
            Rank for convolutional layer LoRA adaptations. Usually lower than linear rank.
            Defaults to 8.

        lora_alpha (int):
            LoRA alpha parameter for scaling. Controls the scaling factor (alpha/rank).
            Defaults to same as rank.

        conv_alpha (int):
            LoRA alpha parameter for conv layers. 
            Defaults to same as conv_rank.

        lora_dropout (float):
            Dropout rate for LoRA layers during training. 
            Defaults to 0.0.

        target_coverage (str):
            Coverage level for LoRA targeting. 
            Choices are "attention", "extended", "full".
            Defaults to "attention".

        enable_conv_lora (bool):
            Whether to include convolutional layers in LoRA adaptation.
            Defaults to False.

        lora_scale (float):
            Global scaling factor for LoRA effects during inference.
            Can be adjusted at runtime. Defaults to 1.0.

        merge_weights (bool):
            Whether to merge LoRA weights into base model permanently.
            If True, creates a merged model. If False, keeps LoRA separate.
            Defaults to False.
    """

    target_modules: List[str] = None
    rank: int = 16
    conv_rank: int = 8
    lora_alpha: int = None  # Will default to rank if not set
    conv_alpha: int = None  # Will default to conv_rank if not set
    lora_dropout: float = 0.0
    target_coverage: Literal["attention", "extended", "full"] = "attention"
    enable_conv_lora: bool = False
    lora_scale: float = 1.0
    merge_weights: bool = False

    def __post_init__(self):
        super().__post_init__()
        
        # Set default alphas if not provided
        if self.lora_alpha is None:
            self.lora_alpha = self.rank
        if self.conv_alpha is None:
            self.conv_alpha = self.conv_rank
            
        # Set default target modules based on coverage
        if self.target_modules is None:
            self.target_modules = self._get_default_target_modules()
        
        # Validation
        assert self.rank > 0, "rank must be positive"
        assert self.conv_rank > 0, "conv_rank must be positive"
        assert 0.0 <= self.lora_dropout <= 1.0, "lora_dropout must be between 0 and 1"
        assert 0.0 <= self.lora_scale, "lora_scale must be non-negative"

    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules based on coverage level"""
        
        if self.target_coverage == "attention":
            return [
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            ]
        elif self.target_coverage == "extended":
            return [
                # Attention
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                # Feed-forward
                "ff.net.0.proj", "ff.net.2",
            ]
        elif self.target_coverage == "full":
            modules = [
                # Attention
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                # Feed-forward
                "ff.net.0.proj", "ff.net.2",
                # Time embedding
                "time_emb_proj",
            ]
            
            # Add conv layers if enabled
            if self.enable_conv_lora:
                modules.extend([
                    "conv1", "conv2", "conv_shortcut",
                    "conv_in", "conv_out",
                    "conv", "downsamplers.0.conv", "upsamplers.0.conv",
                ])
            
            return modules
        else:
            raise ValueError(f"Unknown target_coverage: {self.target_coverage}")
