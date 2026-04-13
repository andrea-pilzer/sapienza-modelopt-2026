#!/bin/bash
# Run this from the login node before starting any experiments.
# Does three things:
#   1. Copies shared assets (SIF + models) to your scratch space
#   2. Initialises the Model-Optimizer git submodule
#   3. Installs course Python packages into your home directory (~/.local)
#      so they are available inside the Singularity container

COURSE_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED=/leonardo/pub/userexternal/apilzer0/assets
ASSETS=$CINECA_SCRATCH/assets
SIF=$ASSETS/pytorch_26.03-py3.sif

# ── 1. Copy shared assets to scratch ─────────────────────────────────────────
if [ -f "$SIF" ] && [ -d "$ASSETS/hf_cache/hub" ]; then
    echo "[setup] Assets already in scratch, skipping copy."
else
    echo "[setup] Copying assets from shared directory (~20 GB, takes 2-3 min)..."
    if [ ! -d "$SHARED" ]; then
        echo "[setup] ERROR: shared assets not found at $SHARED"
        echo "        Ask the instructor to verify the shared path."
        exit 1
    fi
    cp -r "$SHARED" "$ASSETS"
    if [ $? -ne 0 ]; then
        echo "[setup] ERROR: copy failed. Check your scratch quota: du -sh \$CINECA_SCRATCH"
        exit 1
    fi
fi

echo "[setup] Assets OK."

# ── 2. Initialise Model-Optimizer submodule ──────────────────────────────────
if [ -f "$COURSE_DIR/Model-Optimizer/setup.py" ] || [ -f "$COURSE_DIR/Model-Optimizer/pyproject.toml" ]; then
    echo "[setup] Model-Optimizer already present, skipping."
else
    echo "[setup] Initialising Model-Optimizer submodule..."
    git -C "$COURSE_DIR" submodule update --init --recursive
    if [ $? -ne 0 ]; then
        echo "[setup] ERROR: submodule init failed. Check your internet connection."
        exit 1
    fi
fi

echo "[setup] Model-Optimizer ready."

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
