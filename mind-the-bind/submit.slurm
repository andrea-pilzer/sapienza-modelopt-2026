#!/bin/bash
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=4         # number of tasks per node
#SBATCH --cpus-per-task=8           # number of threads per task
#SBATCH --threads-per-core=1        # number of threads per core
#SBATCH --gres=gpu:4                # number of gpus per node
#SBATCH --time 0:00:10              # format: HH:MM:SS
#SBATCH -A tra26_sapai              # billing account
#SBATCH --reservation s_tra_sapai1  # reservation
#SBATCH -p boost_usr_prod
#SBATCH --exclusive

module load nvhpc/24.5 hpcx-mpi/2.19

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#export OMP_PROC_BIND=true

echo "Number of nodes ............. $SLURM_NNODES"
echo "Hosts ....................... $SLURM_NODELIST"
echo "Number of processes ......... $SLURM_NPROCS / SLURM_NPROCS"
echo "Number of tasks ............. $SLURM_NTASKS / SLURM_NTASKS"
echo "Number of cpus per task ..... $SLURM_CPUS_PER_TASK  / SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS ............. $OMP_NUM_THREADS"

echo "SLURM_CPUS_ON_NODE       $SLURM_CPUS_ON_NODE"
echo "SLURM_JOB_CPUS_PER_NODE  $SLURM_JOB_CPUS_PER_NODE"

mpirun -n $SLURM_NPROCS --report-bindings --map-by socket:PE=8 --rank-by core mind-the-bind.x

