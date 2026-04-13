# Sapienza — LLM & Diffusion Model Inference Optimization

**Post-Training Quantization, KV Cache Compression, and Denoising Step Caching**

PhD Course — Sapienza University of Rome  
*2026*

The course covers the principal techniques used in production to make large generative AI
models fast and memory-efficient — without retraining and with minimal accuracy loss.
Hands-on sessions are held on the **Leonardo** cluster at [CINECA](https://www.hpc.cineca.it/systems/hardware/leonardo/),
using NVIDIA A100-SXM-64 GB GPUs.


## Clone Repository

```bash
cd $SCRATCH
git clone --recurse-submodules https://github.com/andrea-pilzer/sapienza-modelopt-2026.git
```

The `--recurse-submodules` flag also clones the **NVIDIA Model Optimizer** library
into `Model-Optimizer/`, required for the quantization experiments in Notebook 2.

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```


## Course Assets (Models, Container Image & Datasets)

The notebooks require pre-downloaded model weights and a Singularity container image.
These are large files (tens of GB) and are **not** stored in this repository.

The course assets (Singularity image, model weights, datasets) are staged at:
```
/leonardo/pub/userexternal/apilzer0/assets/
```

Run the one-time setup script **from the repo root on a login node**. It will copy the assets
to your scratch space and install the required Python packages:

```bash
bash setup.sh
```

This script will:
1. Copy assets from the shared path to `$CINECA_SCRATCH/assets/` (~20 GB, 2–3 min)
2. Initialise the `Model-Optimizer` git submodule
3. Install additional Python packages (`nvidia-modelopt`, `diffusers`) via `pip install --user`
   inside the container — writes only to `~/.local`, does not modify the container image


## Notebooks

| # | Notebook | Topic | Time |
|---|----------|-------|------|
| 01 | `01-CINECA-Leonardo.ipynb` | Leonardo cluster environment: CPUs, GPUs, SLURM, NVLink | 30 min |
| 02 | `02-Model-Optimization.ipynb` | PTQ for LLMs, KV cache compression, cache diffusion, AutoQuantize | ~6 h |

Open the notebooks inside the Singularity container on a compute node (see below).


## Running Notebooks on Leonardo

Hands-on sessions run directly on Leonardo compute nodes via Jupyter.
A double SSH tunnel is required to forward the Jupyter port to your laptop.

**1.** On your laptop, connect to a Leonardo login node:

```bash
ssh USERNAME@login01-ext.leonardo.cineca.it
```

**2.** From the login node, request a GPU compute node (Booster partition):

```bash
srun -t 120 -N 1 --ntasks-per-node 1 --cpus-per-task 4 --mem=64gb \
     --gres=gpu:1 -p boost_usr_prod -A <ACCOUNT> --pty /usr/bin/bash
```

Note the compute node hostname (e.g. `lrdn2191`).

**3.** Start Jupyter inside the Singularity container on the compute node:

```bash
singularity exec --nv --bind $SCRATCH/assets:/workspace/assets --bind $SCRATCH/sapienza-modelopt-2026:/workspace/sapienza-modelopt-2026 $SCRATCH/assets/pytorch_26.03-py3.sif   jupyter notebook --port=9999 --no-browser --ip=0.0.0.0     --notebook-dir=/workspace
```

**4.** On your laptop, open a second terminal and create the SSH tunnel:

Check the node you are using:
```
squeue --me
```
Pick the last bit, something like `lrdnXXXX` and put it in the command below
```bash
ssh -L 9999:localhost:9999 USERNAME@login01-ext.leonardo.cineca.it \
    ssh -L 9999:localhost:9999 -N lrdnXXXX
```

Replace `lrdnXXXX` with the actual compute node name from step 2.

**5.** Open your browser at the URL printed by Jupyter (step 3):

```
http://localhost:9999/?token=<token>
```

For a detailed troubleshooting guide and teacher setup instructions, see [`leonardo-setup.md`](leonardo-setup.md).


## DDP Training Example

`ddp_example/` contains a self-contained multi-GPU distributed training example
using **DistributedDataParallel (DDP)** with MPI, training ResNet-18 on CIFAR-10.

| File | Description |
|------|-------------|
| `ddp_example/ddp_mpi.py` | Training script — supports MPI and SLURM launchers, mixed precision (`GradScaler`), channels-last, and `ZeroRedundancyOptimizer` |
| `ddp_example/ddp_mpi_slurm.sbatch` | SLURM batch script — 4 GPUs on 1 node via `mpirun` |

**Run interactively (2 GPUs, 1 node):**

```bash
# 1. Get an interactive node
salloc -N 1 --ntasks-per-node 2 --cpus-per-task 8 --gres=gpu:2 \
       --mem=32gb -A <ACCOUNT> -p boost_usr_prod -t 01:00:00

# 2. Load modules
module load profile/deeplrn cineca-ai

# 3. Launch
mpirun -np 2 \
  -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
  -x MASTER_PORT=11234 -x PATH \
  -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
  python ddp_example/ddp_mpi.py --num_epochs 5
```

**Submit as a batch job (4 GPUs):**

```bash
# Edit #SBATCH -A to your project account first
sbatch ddp_example/ddp_mpi_slurm.sbatch
```


## GPU Benchmark Utilities

Two CUDA sample binaries are included for hands-on profiling in Notebook 01:

| Binary | What it measures |
|--------|-----------------|
| `bandwidthTest` | Host↔Device and Device↔Device memory bandwidth |
| `p2pBandwidthLatencyTest` | GPU-to-GPU peer-to-peer bandwidth and latency (NVLink) |

Run them on a compute node with:

```bash
./bandwidthTest
./p2pBandwidthLatencyTest
```


## Software Stack

| Component | Version |
|-----------|---------|
| Container | `nvcr.io/nvidia/pytorch:26.03-py3` |
| NVIDIA Model Optimizer | `nvidia-modelopt[all,hf]` (latest) |
| HuggingFace `diffusers` | from `main` branch (system package is broken in container) |
| `transformers` | `>=4.40` |
| Python | 3.10 |


## References

- [NVIDIA Model Optimizer documentation](https://nvidia.github.io/Model-Optimizer/)
- [Top 5 AI Model Optimization Techniques — NVIDIA Developer Blog](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/)
- [SmoothQuant — arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- [AWQ — arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- [PagedAttention / vLLM — arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- [DeepCache — arXiv:2312.00858](https://arxiv.org/abs/2312.00858)
- [HAQ — arXiv:1811.08886](https://arxiv.org/abs/1811.08886)
- [Leonardo cluster user guide](https://wiki.u-gov.it/confluence/pages/viewpage.action?pageId=530321357)
