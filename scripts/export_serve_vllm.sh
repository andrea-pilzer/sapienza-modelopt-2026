#!/bin/bash
# GB200 benchmark: export quantized checkpoints then benchmark with vLLM.
# Requires SLURM + Pyxis/Enroot (srun --container-image).
#
# Usage:
#   HF_CACHE=/path/to/hf_cache bash scripts/export_serve_vllm.sh
#
# Adjust ACCOUNT and PARTITION before running.

ACCOUNT="${ACCOUNT:-<YOUR_ACCOUNT>}"
PARTITION="${PARTITION:-<YOUR_PARTITION>}"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MOUNTS="$REPO_DIR:/workspace,$HF_CACHE:/hf_cache"

PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:26.03-py3"
VLLM_IMAGE="vllm/vllm-openai:latest"

# ── Step 1: Export FP8 and INT4-AWQ checkpoints ───────────────────────────────
srun -N 1 --ntasks=1 --gres=gpu:1 -A "$ACCOUNT" -p "$PARTITION" \
    --container-image="$PYTORCH_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    bash -c "pip install -q --user 'nvidia-modelopt[all,hf]' && \
             HF_HOME=/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
             python scripts/export_quantized.py"

# ── Step 2: Benchmark with vLLM ──────────────────────────────────────────────
# Download vLLM benchmark script if not present
[ -f "$REPO_DIR/scripts/benchmark_throughput.py" ] || \
    curl -sL https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/benchmark_throughput.py \
         -o "$REPO_DIR/scripts/benchmark_throughput.py"

BENCH="python scripts/benchmark_throughput.py --num-prompts 500 --input-len 128 --output-len 128 --dtype bfloat16"

for variant in "bf16:$MODEL:" "fp8:/workspace/exports/fp8:--quantization modelopt" "int4_awq:/workspace/exports/int4_awq:--quantization modelopt"; do
    label="${variant%%:*}"
    rest="${variant#*:}"
    model="${rest%%:*}"
    extra="${rest#*:}"
    echo "--- $label ---"
    srun -N 1 --ntasks=1 --gres=gpu:1 -A "$ACCOUNT" -p "$PARTITION" \
        --container-image="$VLLM_IMAGE" \
        --container-mounts="$MOUNTS" \
        --container-workdir=/workspace \
        bash -c "HF_HOME=/hf_cache HF_HUB_OFFLINE=1 $BENCH --model '$model' $extra"
done
