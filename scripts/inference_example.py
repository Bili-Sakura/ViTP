#!/usr/bin/env python3
"""
Example: Load converted ViTP checkpoint and run inference.

Requires project dependencies: pip install torch safetensors einops timm transformers pillow

First convert a checkpoint:
    python scripts/convert_checkpoint_to_hf.py models/ViTP_ViT_L_300M_rs.safetensors output/ViTP_ViT_L_300M_rs

Then run inference:
    python scripts/inference_example.py output/ViTP_ViT_L_300M_rs --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ViTP"))

import torch
from PIL import Image

# Optional: use torchvision or transformers for preprocessing
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to converted HF-style model dir")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    from internvl.model.internvl_chat import InternVisionModel

    model_path = Path(args.model_path)
    if not (model_path / "config.json").exists():
        print(f"Error: {model_path} does not appear to be a converted HF model (missing config.json)")
        sys.exit(1)

    print(f"Loading model from {model_path}")
    dtype = getattr(torch, args.dtype)
    model = InternVisionModel.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    config = model.config
    image_size = config.image_size
    print(f"  image_size={image_size}, patch_size={config.patch_size}")

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)
        img = Image.open(img_path).convert("RGB")
        if HAS_TORCHVISION:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            pixel_values = transform(img).unsqueeze(0).to(args.device, dtype=dtype)
        else:
            import numpy as np
            arr = np.array(img.resize((image_size, image_size))).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = (arr - mean) / std
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device, dtype=dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        pooled = outputs.pooler_output
        print(f"  Input shape: {pixel_values.shape}")
        print(f"  Pooled output shape: {pooled.shape}")
        print(f"  Pooled norm: {pooled.norm().item():.4f}")
    else:
        # Dummy forward
        dummy = torch.randn(1, 3, image_size, image_size, device=args.device, dtype=dtype)
        with torch.no_grad():
            outputs = model(pixel_values=dummy)
        print(f"  Dummy forward OK, pooled shape: {outputs.pooler_output.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
