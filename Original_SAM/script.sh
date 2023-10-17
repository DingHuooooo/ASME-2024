#!/usr/bin/zsh

### Start of Slurm SBATCH definitions

# Request one node
#SBATCH --nodes=1

# Request two GPUs
#SBATCH --gres=gpu:volta:2

# Ask for 16 CPU cores
#SBATCH --cpus-per-task=16

# Ask for 96 GB memory per CPU
#SBATCH --mem-per-cpu=6G

# Set a walltime limit of 12 hours
#SBATCH --time=24:00:00

# Name the job
#SBATCH --job-name=MPI_JOB

# Declare file where the STDOUT/STDERR outputs will be written
#SBATCH --output=output.%J.txt

### end of Slurm SBATCH definitions

### beginning of executable commands

# Activate the Conda environment
source /home/mr634151/miniconda3/bin/activate
conda activate pytorch4sam

# Run your PyTorch distributed training
python UnetPlusSamPredictor.py
# python -m torch.distributed.launch --nproc_per_node=2 train.py