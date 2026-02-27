@echo off
REM Convert ViTP InternVL 1B models to Hugging Face format.
REM Requires: conda env "sakura" with project dependencies.

cd /d "%~dp0\.."

echo Activating conda env: sakura
call conda activate sakura

echo Converting ViTP_InternVL_1B_general, ViTP_InternVL_1B_med, ViTP_InternVL_1B_rs...
python scripts/convert_internvl_1b.py
