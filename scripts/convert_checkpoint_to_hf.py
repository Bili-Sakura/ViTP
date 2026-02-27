#!/usr/bin/env python3
"""
Convert ViTP checkpoints under models/ to standard Hugging Face Transformers format
for easy pipeline model loading and inference.

Supports:
- Raw ViTP safetensors (ViTP_ViT_L_300M_rs.safetensors, etc.)
- PyTorch .pth/.bin checkpoints (with optional 'module' wrapper)
- Full InternVLChatModel checkpoints (extracts vision_model weights)

Output: config.json + model.safetensors in HF-compatible directory structure,
enabling: InternVisionModel.from_pretrained(output_path)

Usage:
    python scripts/convert_checkpoint_to_hf.py models/ViTP_ViT_L_300M_rs.safetensors output/ViTP_ViT_L_300M_rs
    python scripts/convert_checkpoint_to_hf.py models/ --output-dir output/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add ViTP to path for internvl imports
REPO_ROOT = Path(__file__).resolve().parent.parent
ViTP_PATH = REPO_ROOT / "ViTP"
if str(ViTP_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(ViTP_PATH))

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

# Model config presets (ViT-L 300M from ViTP paper / opencd configs)
VITP_VIT_L_300M_CONFIG = {
    "architectures": ["InternVisionModel"],
    "model_type": "intern_vit_6b",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "patch_size": 16,
    "image_size": 448,
    "num_channels": 3,
    "qkv_bias": True,
    "qk_normalization": False,
    "use_flash_attn": True,
    "hidden_act": "gelu",
    "norm_type": "layer_norm",
    "layer_norm_eps": 1e-6,
    "dropout": 0.0,
    "drop_path_rate": 0.1,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_factor": 0.1,
}


def load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors or PyTorch checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    if path.suffix in (".pth", ".pt", ".bin"):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "module" in ckpt:
            return ckpt["module"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        if isinstance(ckpt, dict):
            return ckpt
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    raise ValueError(f"Unsupported checkpoint format: {path.suffix}")


def extract_vision_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Extract and normalize vision model state dict for InternVisionModel.
    Handles:
    - Raw ViTP: embeddings.*, encoder.* (already correct)
    - Full InternVLChatModel: vision_model.embeddings.*, vision_model.encoder.*
    """
    out = {}
    vision_prefixes = ("vision_model.", "embeddings.", "encoder.")

    for key, value in state_dict.items():
        # Skip non-vision keys (llm, projector, etc.)
        if not any(key.startswith(p) for p in vision_prefixes):
            if key.startswith("vision_model."):
                pass  # will be handled below
            else:
                continue

        new_key = key
        if key.startswith("vision_model."):
            new_key = key.replace("vision_model.", "")
        # Some checkpoints use embedding.proj.* instead of patch_embedding
        new_key = new_key.replace("embedding.proj.", "patch_embedding.")

        out[new_key] = value

    return out


def infer_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    """Infer model config from state dict keys/shapes when possible."""
    config = dict(VITP_VIT_L_300M_CONFIG)

    # Infer hidden_size from class_embedding or patch_embedding
    for key in ("embeddings.class_embedding", "embeddings.patch_embedding.weight"):
        if key in state_dict:
            t = state_dict[key]
            if "class_embedding" in key:
                config["hidden_size"] = int(t.shape[-1])
            elif "patch_embedding" in key:
                config["hidden_size"] = int(t.shape[0])
            break

    # Infer num_hidden_layers from encoder.layers
    layer_keys = [k for k in state_dict if k.startswith("encoder.layers.")]
    if layer_keys:
        indices = set()
        for k in layer_keys:
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers":
                try:
                    indices.add(int(parts[2]))
                except ValueError:
                    pass
        if indices:
            config["num_hidden_layers"] = max(indices) + 1

    # Infer num_attention_heads from qkv weight (prefer 16 for ViT-L)
    for key in state_dict:
        if "attn.qkv.weight" in key:
            dim = state_dict[key].shape[0] // 3  # hidden_size
            for n_heads in [16, 8, 25, 32]:  # prefer 16 for ViT-L
                if dim % n_heads == 0:
                    config["num_attention_heads"] = n_heads
                    break
            break

    # Infer intermediate_size from mlp.fc1
    for key in state_dict:
        if "mlp.fc1.weight" in key:
            config["intermediate_size"] = int(state_dict[key].shape[0])
            break

    # Infer patch_size from patch_embedding kernel
    for key in state_dict:
        if "patch_embedding.weight" in key:
            # shape: (hidden_size, 3, patch_size, patch_size)
            kh, kw = state_dict[key].shape[2], state_dict[key].shape[3]
            if kh == kw:
                config["patch_size"] = int(kh)
            break

    # Infer image_size from position_embedding (num_patches + 1 for cls)
    for key in state_dict:
        if "position_embedding" in key:
            num_pos = state_dict[key].shape[1]  # 1 + num_patches
            num_patches = num_pos - 1
            patch_size = config.get("patch_size", 16)
            grid = int(num_patches ** 0.5)
            if grid * grid == num_patches:
                config["image_size"] = grid * patch_size
            break

    return config


def convert_checkpoint(
    input_path: str | Path,
    output_dir: str | Path,
    config_overrides: dict | None = None,
    infer_config: bool = True,
) -> None:
    """
    Convert a checkpoint to Hugging Face format.

    Args:
        input_path: Path to .safetensors, .pth, .pt, or .bin checkpoint
        output_dir: Output directory for config.json and model.safetensors
        config_overrides: Optional dict to override config values
        infer_config: If True, infer config from state dict; else use preset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {input_path}")
    state_dict = load_state_dict(input_path)

    # Extract vision weights if from full InternVLChatModel
    vision_dict = extract_vision_state_dict(state_dict)
    if not vision_dict:
        # Maybe it's already vision-only but with different structure
        vision_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith("embeddings.") or k.startswith("encoder.")
        }
    if not vision_dict:
        raise ValueError(
            "No vision model weights found. Expected keys starting with "
            "embeddings., encoder., or vision_model."
        )

    print(f"  Found {len(vision_dict)} vision model state dict keys")

    # Build config
    if infer_config:
        config = infer_config_from_state_dict(vision_dict)
    else:
        config = dict(VITP_VIT_L_300M_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    # Save config.json
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Saved config to {config_path}")

    # Save model.safetensors
    model_path = output_dir / "model.safetensors"
    safetensors_save_file(vision_dict, model_path)
    print(f"  Saved weights to {model_path}")

    print(f"\nDone. Load and run inference with:")
    print(f"  import sys")
    print(f"  sys.path.insert(0, 'ViTP')")
    print(f"  from internvl.model.internvl_chat import InternVisionModel")
    print(f"  model = InternVisionModel.from_pretrained('{output_dir}')")
    print(f"  # Forward: outputs = model(pixel_values=images)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViTP checkpoints to Hugging Face Transformers format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input checkpoint path (.safetensors, .pth, .bin) or directory containing checkpoints",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output directory. If input is dir, use --output-dir instead.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory when converting multiple checkpoints",
    )
    parser.add_argument(
        "--no-infer-config",
        action="store_true",
        help="Use preset ViT-L 300M config instead of inferring from state dict",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image_size in config (e.g. 224, 448)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override patch_size in config",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    config_overrides = {}
    if args.image_size is not None:
        config_overrides["image_size"] = args.image_size
    if args.patch_size is not None:
        config_overrides["patch_size"] = args.patch_size

    if input_path.is_file():
        out_dir = args.output or args.output_dir
        if not out_dir:
            # Default: same name as file without extension, under output/
            out_dir = Path("output") / input_path.stem
        convert_checkpoint(
            input_path,
            out_dir,
            config_overrides=config_overrides or None,
            infer_config=not args.no_infer_config,
        )
        return

    # Input is directory: convert all checkpoints under it
    base_out = Path(args.output_dir or args.output or "output")
    exts = {".safetensors", ".pth", ".pt", ".bin"}
    found = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            found.append(p)

    if not found:
        print(f"No checkpoint files found under {input_path}")
        sys.exit(1)

    print(f"Found {len(found)} checkpoint(s)")
    for ckpt in found:
        rel = ckpt.relative_to(input_path)
        out_dir = base_out / rel.parent / ckpt.stem
        try:
            convert_checkpoint(
                ckpt,
                out_dir,
                config_overrides=config_overrides or None,
                infer_config=not args.no_infer_config,
            )
        except Exception as e:
            print(f"  Failed {ckpt}: {e}")
        print()


if __name__ == "__main__":
    main()
