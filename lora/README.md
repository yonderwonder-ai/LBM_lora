# LBM LoRA System

A production-ready LoRA (Low-Rank Adaptation) system for LBM (Latent Diffusion Model) that enables efficient fine-tuning representation and inference.

## 📁 Files Overview

### Core Production Files
- **`extract_lora_native.py`** - Extract LoRA weights from fine-tuned LBM models
- **`inference_native.py`** - Apply LoRA weights during LBM inference
- **`lbm_utils.py`** - Core utilities for LBM model loading and LoRA operations
- **`exact_base.ckpt`** - Exact base model (⚠️ MUST BE CREATED - see Step 0)

### Development & Validation
- **`test_reconstruction_simple.py`** - Validate LoRA reconstruction accuracy
- **`create_exact_base_model.py`** - Generate exact base model from LBM config

## 🚀 Quick Start

### 0. Create Exact Base Model (Required First Step)
```bash
# IMPORTANT: Create the exact base model first (not included in git due to 5GB size)
# Ensure you have the LBM environment activated and dependencies installed
conda activate comfy  # or your LBM environment

python lora/create_exact_base_model.py \
  --config_path "examples/inference/ckpts/relighting/config.yaml" \
  --save_path "lora/checkpoints/exact_base.ckpt" \
  --device "cuda"

# This creates the exact LBM base model matching the fine-tuned model's architecture
# Output: lora/exact_base.ckpt (~4.7GB) + lora/exact_base.safetensors (~4.7GB)
# Note: LPIPS loss excluded (not needed for LoRA, loaded separately during training)
# ⚠️ This step is REQUIRED before any LoRA operations
```

### 1. Extract LoRA from Fine-tuned Model
```bash
# Extract LoRA 
python lora/extract_lora_native.py \
  --base_model_path "lora/exact_base.ckpt" \
  --tuned_model_path "examples/inference/ckpts/relighting/model.safetensors" \
  --save_to "lora/checkpoints/my_lora.safetensors" \
  --rank 32 --conv_rank 16 \ # low rank - efficient but can use heigher rank as well 
  --verbose

```
### 2. Validate LoRA Quality
```bash
python lora/test_reconstruction_simple.py
# Tests reconstruction: Base + LoRA ≈ Fine-tuned
# Reports relative error and success rate
```

### 3. Run Inference with LoRA
```bash
python lora/inference_native.py \
  --base_model_path "lora/checkpoints/exact_base.ckpt" \
  --lora_path "lora/checkpoints/my_lora.safetensors" \
  --source_image "examples/inference/ckpts/relighting/assets/source_image.jpg" \
  --output_path "lora/assets/output.jpg" \
  --lora_scale 1.0 \
  --verbose
```

### 4. Train with LoRA (Optional)
If you want to fine-tune a model using LoRA from scratch, you can use the provided training script.

```bash
# First, customize the configuration file
# open lora/lora_relight_config.yaml and set your dataset paths

# Run LoRA training
python lora/train_lora.py --path_config lora/lora_relight_config.yaml
```

### 5. Advanced LoRA Fine-Tuning
You can control exactly which LoRA layers are trained by editing the `lora/lora_relight_config.yaml` file. This is useful for advanced fine-tuning, such as training only attention layers while keeping others frozen.

1.  **Generate Layer Names**: First, run the reconstruction test script. This will create a file named `lora/lora_layer_names.txt` containing all possible LoRA layer names.
    ```bash
    python lora/test_reconstruction_simple.py
    ```

2.  **Configure Training**: Open `lora/lora_relight_config.yaml` and find the `trainable_lora_modules` section. Copy the names of the layers you want to train from `lora_layer_names.txt` into this list. You can use partial names to select groups of layers (e.g., `"attn1"` will select all self-attention layers).

    **Example**: To train only the self-attention (`attn1`) and feed-forward (`ff.net`) layers:
    ```yaml
    trainable_lora_modules:
      - "attn1"
      - "ff.net"
    ```

## 🔧 Key Parameters

### LoRA Extraction
- **`--rank`** - Rank for linear layers (32=efficient, 9999=full rank)
- **`--conv_rank`** - Rank for conv layers (16=efficient, 9999=full rank)
- **`--lora_scale`** - Scaling factor during inference (0.0-2.0, default=1.0)

### Quality vs Efficiency
- **Low Rank (32/16)**: Small files, fast inference, some quality loss
- **Full Rank (9999)**: Large files, slower, perfect reconstruction

## 📊 Understanding Results

### Relative Error Interpretation
- **< 5%**: Excellent LoRA quality
- **5-50%**: Good quality, minor artifacts  
- **50-100%**: Poor quality, noticeable degradation
- **> 100%**: More noise than signal (increase rank)

### Success Metrics
- **792 modules**: Complete coverage of all LBM denoiser layers
- **100% success rate**: All LoRA modules applied correctly
- **0.38% avg error**: Nearly perfect reconstruction (full rank)

## 🎯 Production Usage

### For Deployment (Recommended)
```bash
# Extract efficient LoRA
python lora/extract_lora_native.py --rank 64 --conv_rank 32

# Apply during inference  
python lora/inference_native.py --lora_scale 1.0
```

### For Research/Quality Analysis
```bash
# Extract perfect LoRA
python lora/extract_lora_native.py --rank 9999 --conv_rank 9999

# Validate quality
python lora/test_reconstruction_simple.py
```

## 🔬 Technical Details

- **Native LBM Naming**: Uses exact LBM state dict keys (no conversion bugs)
- **Full Layer Coverage**: All 792 denoiser weight layers supported
- **Exact Base Model**: Matches LBM training initialization exactly
- **Mathematical Validation**: SVD-based extraction with verified reconstruction

## ⚡ System Requirements

- LBM environment activated (`conda activate comfy`)
- PyTorch with CUDA support
- safetensors for efficient model storage
- LBM dependencies (lpips, diffusers, etc.)
- Sufficient GPU memory for model loading
- ~5GB disk space for exact base model

## 📝 Important Notes

### Git Repository
- **`exact_base.ckpt`** is excluded from git (5GB file size)
- **MUST run Step 0** to create base model before using LoRA system
- Only source code and small config files are version controlled

**Recommended .gitignore entries:**
```gitignore
# Exclude large model files
lora/exact_base.ckpt
*.safetensors
*.ckpt
*.jpg
*.png
```

### File Sizes
- `exact_base.ckpt` + `exact_base.safetensors`: ~4.7GB each (LBM base model, both formats)
- LoRA files: 100MB-2GB (depending on rank)
- Small scripts: <50KB each

### First-Time Setup
1. Clone repository (contains only source code)
2. Activate LBM environment (`conda activate comfy`)
3. Run Step 0 to create base model
4. Proceed with LoRA extraction/inference

### LoRA Training Compatibility
- Base model contains core LBM architecture (UNet, VAE, conditioners)
- LPIPS loss is loaded separately during training (standard practice)
- Both .ckpt and .safetensors formats available for different use cases
- Perfect for LoRA training workflows

---

*This LoRA system achieves mathematically verified reconstruction with 100% module coverage and production-ready performance.*
