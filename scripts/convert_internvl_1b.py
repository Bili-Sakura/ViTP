#!/usr/bin/env python3
"""
Convert ViTP InternVL 1B safetensors to Hugging Face format.

Converts:
  - models/ViTP_InternVL_1B_general.safetensors
  - models/ViTP_InternVL_1B_med.safetensors
  - models/ViTP_InternVL_1B_rs.safetensors

Output: config.json + model.safetensors in output/ViTP_InternVL_1B_{general,med,rs}/

Usage:
    conda activate sakura
    python scripts/convert_internvl_1b.py

Or use the shell script:
    bash scripts/convert_internvl_1b.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import convert_checkpoint from sibling module
import importlib.util
_convert_spec = importlib.util.spec_from_file_location(
    "convert_checkpoint_to_hf",
    REPO_ROOT / "scripts" / "convert_checkpoint_to_hf.py",
)
_convert_mod = importlib.util.module_from_spec(_convert_spec)
_convert_spec.loader.exec_module(_convert_mod)
convert_checkpoint = _convert_mod.convert_checkpoint

INTERNVL_1B_MODELS = [
    "ViTP_InternVL_1B_general.safetensors",
    "ViTP_InternVL_1B_med.safetensors",
    "ViTP_InternVL_1B_rs.safetensors",
]


def main() -> None:
    models_dir = REPO_ROOT / "models"
    output_dir = REPO_ROOT / "output"

    for filename in INTERNVL_1B_MODELS:
        input_path = models_dir / filename
        stem = input_path.stem  # e.g. ViTP_InternVL_1B_general
        out_path = output_dir / stem

        if not input_path.exists():
            print(f"Skipping (not found): {input_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Converting: {filename}")
        print(f"{'='*60}")
        try:
            convert_checkpoint(input_path, out_path, infer_config=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print(f"\n{'='*60}")
    print("All conversions complete.")
    print(f"Output directories: {output_dir}/ViTP_InternVL_1B_{{general,med,rs}}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
