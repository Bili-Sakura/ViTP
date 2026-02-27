#!/usr/bin/env python3
"""
Add modeling code and auto_map to existing output folders so models can be loaded
without the ViTP repo (using AutoModel + trust_remote_code=True).

Usage:
    python scripts/add_modeling_code_to_output.py
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
INTERNVL_CHAT = REPO_ROOT / "ViTP" / "internvl" / "model" / "internvl_chat"


def main():
    model_dirs = [
        d for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists() and d.name != "__pycache__"
    ]
    if not model_dirs:
        print(f"No model directories found in {OUTPUT_DIR}")
        return

    for model_dir in model_dirs:
        print(f"Updating {model_dir.name}...")

        # Add auto_map to config.json
        config_path = model_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        if "auto_map" not in config:
            config["auto_map"] = {
                "AutoConfig": "configuration_intern_vit.InternVisionConfig",
                "AutoModel": "modeling_intern_vit.InternVisionModel",
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"  Added auto_map to config.json")

        # Copy modeling files
        for name in ("configuration_intern_vit.py", "modeling_intern_vit.py"):
            src = INTERNVL_CHAT / name
            if src.exists():
                shutil.copy2(src, model_dir / name)
                print(f"  Copied {name}")

    print(f"\nDone. {len(model_dirs)} model(s) updated.")


if __name__ == "__main__":
    main()
