#!/bin/bash
# Run this from the login node before starting any experiments.
# Does three things:
#   1. Checks that shared assets (SIF + models) are in place
#   2. Clones the Model-Optimizer repository (Z-Image branch)
#   3. Installs course Python packages into your home directory (~/.local)
#      so they are available inside the Singularity container

COURSE_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS=/leonardo_scratch/large/userexternal/$USER/assets
SIF=$ASSETS/pytorch_26.03-py3.sif

# ── 1. Check assets ──────────────────────────────────────────────────────────
echo "[setup] Checking assets..."

if [ ! -f "$SIF" ]; then
    echo "[setup] ERROR: Singularity image not found at $SIF"
    echo "        Did you complete Step 2 (copy shared assets)?"
    exit 1
fi

if [ ! -d "$ASSETS/hf_cache/hub" ]; then
    echo "[setup] ERROR: model cache not found at $ASSETS/hf_cache/hub"
    echo "        Did you complete Step 2 (copy shared assets)?"
    exit 1
fi

echo "[setup] Assets OK."

# ── 2. Clone Model-Optimizer ─────────────────────────────────────────────────
if [ -d "$COURSE_DIR/Model-Optimizer" ]; then
    echo "[setup] Model-Optimizer already cloned, skipping."
else
    echo "[setup] Cloning Model-Optimizer (Z-Image branch)..."
    git clone --depth=1 \
        --branch feat/zimage-quantization-support \
        https://github.com/andrea-pilzer/Model-Optimizer.git \
        "$COURSE_DIR/Model-Optimizer"
    if [ $? -ne 0 ]; then
        echo "[setup] ERROR: git clone failed. Check your internet connection."
        exit 1
    fi
    echo "[setup] Model-Optimizer cloned."
fi

# ── 3. Install Python packages ───────────────────────────────────────────────
echo "[setup] Installing packages (this takes ~5 minutes)..."

singularity exec "$SIF" bash -c '
    pip install --user --quiet \
        "nvidia-modelopt[all,hf]" \
        sentencepiece \
        nbconvert ipykernel datasets && \
    pip install --user --quiet \
        "git+https://github.com/huggingface/diffusers@main"
'

if [ $? -eq 0 ]; then
    echo "[setup] Done. Environment is ready."
else
    echo "[setup] ERROR: package installation failed. Check your internet connection and retry."
    exit 1
fi
