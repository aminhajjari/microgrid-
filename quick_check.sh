#!/bin/bash
#SBATCH --job-name=qcheck
#SBATCH --account=def-arashmoh
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/gkianfar/scratch/Amin/MSH/output/logs/qcheck_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/MSH/output/logs/qcheck_%j.err

set -e

echo "=== quick_check on $(hostname) — $(date) ==="

module purge
module load StdEnv/2023 python/3.10.13 cuda/12.2

CODE_PATH="/home/gkianfar/scratch/Amin/MSH/microgrid-"
VENV_PATH="/home/gkianfar/scratch/Amin/MSH/microgridvenv"

cd "$CODE_PATH"
source "$VENV_PATH/bin/activate"
echo "✓ venv: $VIRTUAL_ENV"

# limit thread thrashing on the allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "--- 0. import check ---"
python -c "from metrics import rur, degradation, convergence_ep, EpisodeTimer, build_table4_row, print_table4; print('imports OK')"

echo ""
echo "--- 1. quick_check.py ---"
python quick_check.py

echo ""
echo "--- 2. demo.py smoke test (60 ep) ---"
python demo.py

echo ""
echo "=== done — $(date) ==="
