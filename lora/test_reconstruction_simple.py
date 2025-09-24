"""
Simple LoRA Reconstruction Test

Direct state dict comparison without model loading issues.
Tests if: Exact_Base + LoRA ≈ Fine-tuned
"""

import torch
from safetensors.torch import load_file

def test_lora_reconstruction():
    """Simple reconstruction test using LBM native naming directly"""
    
    print("🧪 COMPLETE LoRA RECONSTRUCTION TEST (LBM Native Naming)")
    print("=" * 65)
    
    device = "cpu"
    
    # Load exact base model state dict
    print("📂 Loading exact base model...")
    exact_base = torch.load('lora/checkpoints/exact_base.ckpt', map_location=device, weights_only=False)
    base_state = exact_base['model_state_dict']
    print(f"✓ Loaded exact base: {len(base_state)} parameters")
    
    # Load fine-tuned model state dict  
    print("📂 Loading fine-tuned model...")
    tuned_state = load_file('examples/inference/ckpts/relighting/model.safetensors', device=device)
    print(f"✓ Loaded fine-tuned: {len(tuned_state)} parameters")
    
    # Load LoRA weights (FULL RANK TEST)
    print("📂 Loading LoRA weights...")
    lora_weights = load_file('lora/checkpoints/my_lora.safetensors', device=device)
    lora_modules = len([k for k in lora_weights.keys() if k.endswith('.lora_up.weight')])
    print(f"✓ Loaded LoRA: {lora_modules} modules")
    
    print()
    print("🧮 Testing LoRA reconstruction using LBM native keys...")
    
    # Group LoRA weights by base key (LBM native naming)
    lora_modules = {}
    for key, weight in lora_weights.items():
        if ".lora_up.weight" in key:
            base_key = key.replace(".lora_up.weight", "")
            if base_key not in lora_modules:
                lora_modules[base_key] = {}
            lora_modules[base_key]["up"] = weight
        elif ".lora_down.weight" in key:
            base_key = key.replace(".lora_down.weight", "")
            if base_key not in lora_modules:
                lora_modules[base_key] = {}
            lora_modules[base_key]["down"] = weight
        elif ".alpha" in key:
            base_key = key.replace(".alpha", "")
            if base_key not in lora_modules:
                lora_modules[base_key] = {}
            lora_modules[base_key]["alpha"] = weight
    
    # Save the identified LoRA layer names to a file for reference
    lora_layer_names_path = "lora/lora_layer_names.txt"
    try:
        with open(lora_layer_names_path, "w") as f:
            f.write(
                "# A list of all LoRA module names found in the model.\n"
                "# Use these names (or parts of them) as patterns in `trainable_lora_modules` in your config file.\n\n"
            )
            for key in sorted(lora_modules.keys()):
                f.write(f"{key}\n")
        print(f"✅ Saved LoRA layer names to: {lora_layer_names_path}")
    except Exception as e:
        print(f"⚠️ Could not save LoRA layer names: {e}")

    print(f"📊 Testing {len(lora_modules)} LoRA modules using native LBM keys...")
    
    # Test reconstruction for each LoRA module
    reconstruction_errors = []
    perfect_matches = 0
    failed_matches = 0
    
    for lora_key, lora_data in lora_modules.items():
        # Skip incomplete modules
        if "up" not in lora_data or "down" not in lora_data:
            continue
        
        # Convert LoRA key to target weight key
        target_key = f"{lora_key}.weight"
        
        # Check if target exists in both base and tuned models
        if target_key not in base_state or target_key not in tuned_state:
            failed_matches += 1
            continue
        
        # Get tensors
        base_weight = base_state[target_key]
        tuned_weight = tuned_state[target_key] 
        lora_up = lora_data["up"]
        lora_down = lora_data["down"]
        alpha = lora_data.get("alpha", torch.tensor(32.0))
        
        # Calculate ground truth delta
        expected_delta = tuned_weight - base_weight
        
        # Calculate LoRA reconstruction (handle all tensor shapes)
        try:
            lora_delta = None
            rank = None
            
            # Handle Linear layers (2D tensors)
            if len(lora_up.shape) == 2 and len(lora_down.shape) == 2:
                if lora_up.shape[1] == lora_down.shape[0]:  # [out, rank] @ [rank, in]
                    lora_delta = lora_up @ lora_down
                    rank = lora_up.shape[1]
                elif lora_up.shape[0] == lora_down.shape[1]:  # Swapped order
                    lora_delta = lora_down @ lora_up
                    rank = lora_up.shape[0]
            
            # Handle Conv layers (4D tensors) - reshape to 2D, multiply, reshape back
            elif len(lora_up.shape) == 4 or len(lora_down.shape) == 4:
                # Flatten conv tensors for matrix multiplication
                up_2d = lora_up.flatten(1) if len(lora_up.shape) == 4 else lora_up
                down_2d = lora_down.flatten(1) if len(lora_down.shape) == 4 else lora_down
                
                if up_2d.shape[1] == down_2d.shape[0]:
                    lora_delta = (up_2d @ down_2d).reshape(base_weight.shape)
                    rank = up_2d.shape[1]
                elif up_2d.shape[0] == down_2d.shape[1]:
                    lora_delta = (down_2d @ up_2d).reshape(base_weight.shape)
                    rank = up_2d.shape[0]
            
            # Apply scaling and calculate errors
            if lora_delta is not None and rank is not None:
                alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                
                # For full rank LoRA, don't apply additional alpha scaling
                # (SVD decomposition already captures full information)
                matrix_min_dim = min(base_weight.shape)
                if rank >= matrix_min_dim:
                    # Full rank - no additional scaling needed
                    scale_factor = 1.0
                else:
                    # Low rank - apply standard LoRA scaling
                    scale_factor = alpha_val / rank
                
                lora_delta = lora_delta * scale_factor
                
                reconstruction_error = torch.mean(torch.abs(expected_delta - lora_delta)).item()
                relative_error = reconstruction_error / (torch.mean(torch.abs(expected_delta)).item() + 1e-8)
                
                reconstruction_errors.append({
                    'lora_key': lora_key,
                    'target_key': target_key,
                    'error': reconstruction_error,
                    'relative_error': relative_error,
                    'expected_mag': torch.mean(torch.abs(expected_delta)).item(),
                    'lora_mag': torch.mean(torch.abs(lora_delta)).item(),
                    'rank': rank,
                    'shapes': f"up:{list(lora_up.shape)}, down:{list(lora_down.shape)}"
                })
                
                if reconstruction_error < 0.001:
                    perfect_matches += 1
            else:
                failed_matches += 1
                
        except Exception as e:
            print(f"❌ Error processing {lora_key}: {e}")
            failed_matches += 1
            continue
    
    print()
    print("🎯 COMPLETE LoRA RECONSTRUCTION RESULTS:")
    print("-" * 55)
    
    total_attempted = len(lora_modules)
    total_successful = len(reconstruction_errors)
    
    print(f"📊 LoRA modules found: {total_attempted}")
    print(f"📊 Successfully tested: {total_successful}")
    print(f"📊 Failed to match: {failed_matches}")
    print(f"📊 Success rate: {total_successful/total_attempted*100:.1f}%")
    
    if reconstruction_errors:
        avg_error = sum(e['error'] for e in reconstruction_errors) / total_successful
        avg_relative_error = sum(e['relative_error'] for e in reconstruction_errors) / total_successful
        
        print(f"📊 Perfect matches (error < 0.001): {perfect_matches}/{total_successful} ({perfect_matches/total_successful*100:.1f}%)")
        print()
        print(f"📏 Average reconstruction error: {avg_error:.8f}")
        print(f"📏 Average relative error: {avg_relative_error:.2%}")
        print()
        
        # Show best and worst reconstructions
        best = min(reconstruction_errors, key=lambda x: x['relative_error'])
        worst = max(reconstruction_errors, key=lambda x: x['relative_error'])
        
        print(f"✅ Best reconstruction:  {best['relative_error']:.2%} error ({best['lora_key']})")
        print(f"❌ Worst reconstruction: {worst['relative_error']:.2%} error ({worst['lora_key']})")
        print()
        
        # Show some example LoRA keys to verify native naming
        print("🔍 SAMPLE LoRA KEYS (Native LBM Naming):")
        for i, result in enumerate(reconstruction_errors[:3]):
            print(f"   {i+1}. {result['lora_key']} → {result['target_key']}")
            print(f"      Shapes: {result['shapes']}, Rank: {result['rank']}, Error: {result['relative_error']:.2%}")
        
        print()
        
        # Quality assessment
        if avg_relative_error < 0.05:
            print("🎉 EXCELLENT: LoRA reconstruction is nearly perfect!")
            print("   The LoRA math and native naming are working correctly.")
        elif avg_relative_error < 0.20:
            print("✅ GOOD: LoRA reconstruction is working well.")  
            print("   Some information loss due to rank limitation.")
        else:
            print("⚠️  HIGH RELATIVE ERROR: LoRA reconstruction shows high noise-to-signal ratio.")
            print("   This may indicate rank is too low for the subtle changes in this model.")
    
    else:
        print("❌ No modules tested successfully")
        print("   Check LoRA file format and key naming.")


if __name__ == "__main__":
    test_lora_reconstruction()
