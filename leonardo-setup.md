# For the Teacher

## TODO before distributing
- [ ] Replace `<REPO_URL>` in Step 1 with the actual GitHub repo URL
- [ ] Replace `<SHARED_ASSETS_PATH>` in Step 2 with the actual cluster path where assets are staged (use `$WORK` so all students in `cin_extern02_1` can access it)
- [ ] Set read permissions on the shared assets directory: `chmod -R o+rX <SHARED_ASSETS_PATH>`

/leonardo_scratch/large/userexternal/apilzer0/pytorch_26.03-py3.sif                                                      
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct                         
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0             
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/hub/models--PixArt-alpha--PixArt-XL-2-512x512                     
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/hub/datasets--cnn_dailymail                                       
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/hub/datasets--Gustavosta--Stable-Diffusion-Prompts                
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/datasets/cnn_dailymail                                            
/leonardo_scratch/large/userexternal/apilzer0/hf_cache/datasets/Gustavosta___stable-diffusion-prompts

# Environment Setup on CINECA Leonardo
## PhD Course — LLM & Diffusion Model Inference Optimization

---

## Background

Leonardo has two types of nodes: **login nodes** (where you land when you SSH in) and
**compute nodes** (where GPU jobs run). Only login nodes have internet access — compute
nodes are network-isolated. This means all model weights and Python packages must be
staged on the shared filesystem *before* running any experiments.

To keep the software environment reproducible, all Python code runs inside a
**Singularity container** (a portable, read-only Linux image with CUDA and PyTorch
pre-installed). You will install the course-specific packages on top of it once using
`pip install --user`, which places them in your home directory and makes them
available inside the container automatically.

---

## Step 1 — Clone the Repository

From the login node, clone the course repository into your scratch space:

```bash
cd /leonardo_scratch/large/userexternal/$USER
git clone <REPO_URL> course
```

You should now have:
```
/leonardo_scratch/large/userexternal/$USER/course/
├── Model Optimization.ipynb
├── setup.sh
└── ...
```

---

## Step 2 — Copy Shared Assets

The course assets (Singularity image + pre-downloaded model weights + datasets) are
stored at a shared path on the cluster so you do not need to download them yourself.

Copy them to your scratch space:

```bash
cp -r <SHARED_ASSETS_PATH> /leonardo_scratch/large/userexternal/$USER/assets
```

This copies ~20 GB and takes about 2–3 minutes. When done, verify the structure:

```bash
ls /leonardo_scratch/large/userexternal/$USER/assets/
```

Expected output:
```
hf_cache/
pytorch_26.03-py3.sif
```

```bash
ls /leonardo_scratch/large/userexternal/$USER/assets/hf_cache/hub/
```

Expected output:
```
models--Qwen--Qwen2.5-1.5B-Instruct
models--stabilityai--stable-diffusion-xl-base-1.0
models--PixArt-alpha--PixArt-XL-2-512x512
```

> If anything is missing, ask the instructor — do not try to download models yourself,
> compute nodes cannot reach the internet.

---

## Step 3 — Run Setup

Run the setup script **from the login node** (not inside a job). It will:
- verify the assets from Step 2 are in place
- clone the Model-Optimizer repository (Z-Image branch)
- install the course Python packages into your home directory

```bash
cd /leonardo_scratch/large/userexternal/$USER/course
bash setup.sh
```

The script takes about **5 minutes** (dominated by pip downloads). When it finishes
you should see:

```
[setup] Assets OK.
[setup] Model-Optimizer cloned.
[setup] Done. Environment is ready.
```

Verify everything is in place:

```bash
# Packages
singularity exec \
  /leonardo_scratch/large/userexternal/$USER/assets/pytorch_26.03-py3.sif \
  python3 -c "import modelopt; import diffusers; print('packages OK')"

# Model-Optimizer code
ls /leonardo_scratch/large/userexternal/$USER/course/Model-Optimizer/examples/diffusers/
```

Expected output: `packages OK` and a listing containing `cache_diffusion/`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: modelopt` after setup | `setup.sh` did not complete | Re-run `bash setup.sh` and wait for `[setup] Done. Environment is ready.` |
| `ModuleNotFoundError: diffusers` | GitHub diffusers install failed | Check internet on login node: `curl -s https://github.com` — then re-run `bash setup.sh` |
| `ModuleNotFoundError: cache_diffusion` | Model-Optimizer not cloned | Check `ls course/Model-Optimizer/` — if missing, re-run `bash setup.sh` |
| `ls assets/hf_cache/hub/` shows missing models | Copy in Step 2 was incomplete | Re-run the `cp -r` command |
| `No space left on device` during copy | Your scratch quota is full | Run `du -sh /leonardo_scratch/large/userexternal/$USER/` to check usage |
