#!/usr/bin/env bash
# Convert ViTP InternVL 1B models to Hugging Face format.
# Requires: conda env "sakura" with project dependencies.

set -e
cd "$(dirname "$0")/.."

echo "Activating conda env: sakura"
eval "$(conda shell.bash hook)"
conda activate sakura

echo "Converting ViTP_InternVL_1B_general, ViTP_InternVL_1B_med, ViTP_InternVL_1B_rs..."
python scripts/convert_internvl_1b.py
