#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=arw_test
#SBATCH --output=/home/gkianfar/scratch/Amin/MSH/output/logs/arw_test_%j.out

module load StdEnv/2023
source /lustre07/scratch/gkianfar/Amin/MSH/microgridvenv/bin/activate
cd /lustre07/scratch/gkianfar/Amin/MSH/microgrid-

echo "=== ARW stability test ==="
python test_arw_stability.py || exit 1
echo "=== quick_check ==="
python quick_check.py || exit 1
echo "=== demo (60 ep smoke) ==="
python demo.py
echo "=== ALL TESTS DONE ==="
