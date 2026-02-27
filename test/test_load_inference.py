"""
Test load and inference of converted ViTP HF-style models.

Run from project root:
    python test/test_load_inference.py
    pytest test/test_load_inference.py -v

Requires: pip install torch safetensors einops timm transformers peft

Prerequisite: Convert a checkpoint first:
    python scripts/convert_checkpoint_to_hf.py models/ViTP_ViT_L_300M_med.safetensors output/ViTP_ViT_L_300M_med
"""

import os
import sys
import warnings
from pathlib import Path

# Disable meta device to avoid "Tensor.item() cannot be called on meta tensors"
os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")
# Reduce noise from flash-attn warnings
warnings.filterwarnings("ignore", message="FlashAttention")
warnings.filterwarnings("ignore", message="flash-attention")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ViTP"))


def _load_model(model_dir: Path, torch_dtype="float32"):
    """Load model avoiding meta device issues."""
    from internvl.model.internvl_chat.configuration_intern_vit import InternVisionConfig
    from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel

    import torch
    from safetensors.torch import load_file as load_safetensors

    config = InternVisionConfig.from_pretrained(str(model_dir))
    # Create model on CPU directly (bypass from_pretrained meta device)
    model = InternVisionModel(config)
    state_dict = load_safetensors(str(model_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    model = model.to(dtype)
    return model


def test_load_converted_model():
    """Test loading converted HF-style model from output/."""
    model_dir = REPO_ROOT / "output" / "ViTP_ViT_L_300M_med"
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Converted model not found at {model_dir}. "
            "Run: python scripts/convert_checkpoint_to_hf.py models/ViTP_ViT_L_300M_med.safetensors output/ViTP_ViT_L_300M_med"
        )

    model = _load_model(model_dir)
    assert model.config.hidden_size == 1024
    assert model.config.num_hidden_layers == 24
    assert model.config.num_attention_heads == 16
    assert model.config.image_size == 448
    assert model.config.patch_size in (14, 16)  # ViT-L varies by checkpoint
    print("  Load OK: config matches ViT-L 300M")


def test_inference_dummy():
    """Test dummy forward pass."""
    import torch

    model_dir = REPO_ROOT / "output" / "ViTP_ViT_L_300M_med"
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Converted model not found at {model_dir}. Run conversion first."
        )

    model = _load_model(model_dir, torch.float32)
    model.eval()

    image_size = model.config.image_size
    dummy = torch.randn(1, 3, image_size, image_size)  # batch=1 for faster CPU test

    with torch.no_grad():
        outputs = model(pixel_values=dummy)

    patch_size = model.config.patch_size
    assert outputs.pooler_output.shape == (1, 1024)
    assert outputs.last_hidden_state.shape == (1, (image_size // patch_size) ** 2 + 1, 1024)
    print(f"  Inference OK: pooled {outputs.pooler_output.shape}, last_hidden {outputs.last_hidden_state.shape}")


def test_inference_cpu():
    """Test inference on CPU (no CUDA required)."""
    import torch

    model_dir = REPO_ROOT / "output" / "ViTP_ViT_L_300M_med"
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Converted model not found at {model_dir}.")

    model = _load_model(model_dir, torch.float32)
    model.eval()

    image_size = model.config.image_size
    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        out = model(pixel_values=dummy)

    assert out.pooler_output.shape == (1, 1024)
    print("  CPU inference OK")


if __name__ == "__main__":
    print("Testing load and inference...")
    test_load_converted_model()
    test_inference_dummy()
    test_inference_cpu()
    print("All tests passed.")
