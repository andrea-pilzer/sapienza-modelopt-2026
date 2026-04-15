"""
Export Qwen2.5-1.5B-Instruct to FP8 and INT4-AWQ checkpoints using NVIDIA ModelOpt.
Run inside the PyTorch container (Step 1 of run_gb200.sh).

Outputs:
    /workspace/exports/fp8/
    /workspace/exports/int4_awq/
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader
from modelopt.torch.export import export_hf_checkpoint

MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "/workspace/exports"
CALIB_SIZE = 64

FORMATS = {
    "fp8":      mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
).eval()

print(f"Building calibration dataloader ({CALIB_SIZE} samples) ...")
dataloader = get_dataset_dataloader(
    dataset_name="cnn_dailymail",
    tokenizer=tokenizer,
    batch_size=4,
    num_samples=CALIB_SIZE,
    device="cuda",
)
forward_loop = create_forward_loop(dataloader=dataloader)

for fmt, cfg in FORMATS.items():
    out = os.path.join(OUTPUT_DIR, fmt)
    if os.path.isdir(out):
        print(f"[{fmt}] already exported, skipping.")
        continue

    print(f"[{fmt}] Quantizing ...")
    mtq.quantize(model, cfg, forward_loop)

    print(f"[{fmt}] Exporting to {out} ...")
    export_hf_checkpoint(model, out)

    # restore for next format
    print(f"[{fmt}] Reloading base model for next format ...")
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

print("Export complete.")
