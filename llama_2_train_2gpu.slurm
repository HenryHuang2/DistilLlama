#!/bin/bash

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=2

# The name of the job:
#SBATCH --job-name="train_2gpu"

# The project ID which this job should run under:
#SBATCH --account="punim1257"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# The amount of memory in gigabytes per CPU:
#SBATCH --mem-per-cpu=32G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=7-0:0:0

# Use this email address:
#SBATCH --mail-user=henry.huang@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

#SBATCH --output=llama_2_train_2gpu.out

# Check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Load necessary modules:
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load foss/2022a
module load CUDA/11.7.0
module load UCX-CUDA/1.13.1-CUDA-11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

# Activate virtual environment
source ~/venvs/venv-3.10.4/bin/activate

# Set environment variables for distributed training:
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

# Execute the Python script:
srun python3 /data/gpfs/projects/punim1257/Knowledge-Distillation/llama-main/main.py

## DO NOT ADD/EDIT BEYOND THIS LINE ##
## Job monitor command to list the resource usage
my-job-stats -a -n -s
