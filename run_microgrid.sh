#!/bin/bash
#SBATCH --job-name=hma_drl_microgrid
#SBATCH --account=def-arashmoh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/gkianfar/scratch/Amin/MSH/output/logs/hma_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/MSH/output/logs/hma_%j.err

set -e

echo "========================================="
echo "HMA-DRL MICROGRID ENERGY MANAGEMENT"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================="

# ==============================
# LOAD MODULES
# ==============================
module purge
module load gcc python/3.10 cuda/12.6
echo "✓ Modules loaded"

# ==============================
# PATHS
# ==============================
CODE_PATH="/home/gkianfar/scratch/Amin/MSH/microgrid-"
OUTPUT_BASE="/home/gkianfar/scratch/Amin/MSH/output"
VENV_PATH="/home/gkianfar/scratch/Amin/MSH/microgridvenv"

mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/plots
mkdir -p $OUTPUT_BASE/checkpoints

cd $CODE_PATH || exit 1

# ==============================
# VIRTUAL ENV
# ==============================
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment …"
    python -m venv $VENV_PATH
fi
source $VENV_PATH/bin/activate
echo "✓ Environment: $VIRTUAL_ENV"

# ==============================
# DEPENDENCIES
# ==============================
echo "=== Checking / Installing Dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet numpy matplotlib gymnasium
echo "✓ Dependencies ready"

# ==============================
# ENV VARIABLES
# ==============================
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ==============================
# GPU INFO
# ==============================
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ==============================
# VERIFY CODE FILES
# ==============================
echo "=== Checking Code Files ==="
for f in microgrid_env.py agents.py hma_drl.py metrics.py train.py demo.py; do
    if [ -f "$CODE_PATH/$f" ]; then
        echo "✅ $f found"
    else
        echo "❌ $f MISSING"
        exit 1
    fi
done
echo ""

# ==============================
# HELPER — training run
# ==============================
run_train () {
    local METHOD=$1
    local SCENARIO=$2
    local EPISODES=$3
    local LOG="$OUTPUT_BASE/logs/${METHOD}_${SCENARIO}_${SLURM_JOB_ID}.log"

    echo "--------------------------------------"
    echo "Training: method=$METHOD  scenario=$SCENARIO  episodes=$EPISODES"
    echo "Log: $LOG"
    echo "--------------------------------------"

    python train.py \
        --method   $METHOD   \
        --scenario $SCENARIO \
        --episodes $EPISODES \
        --batch    256       \
        --device   cuda      \
        2>&1 | tee "$LOG"

    EXIT_CODE=${PIPESTATUS[0]}
    [ $EXIT_CODE -ne 0 ] && { echo "❌ ERROR: $METHOD/$SCENARIO failed"; exit $EXIT_CODE; }

    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
    echo "✓ Done: $METHOD / $SCENARIO"
    echo ""
}

# ==============================
# HELPER — stress evaluation
# Loads the pre-trained normal checkpoint and evaluates on a stress scenario
# (Section 3.4 of the paper)
# ==============================
run_stress () {
    local METHOD=$1
    local SCENARIO=$2
    local LOG="$OUTPUT_BASE/logs/${METHOD}_${SCENARIO}_stress_${SLURM_JOB_ID}.log"

    echo "--------------------------------------"
    echo "Stress eval: method=$METHOD  scenario=$SCENARIO"
    echo "Log: $LOG"
    echo "--------------------------------------"

    python train.py \
        --method    $METHOD   \
        --scenario  $SCENARIO \
        --batch     256       \
        --device    cuda      \
        --eval-only           \
        2>&1 | tee "$LOG"

    EXIT_CODE=${PIPESTATUS[0]}
    [ $EXIT_CODE -ne 0 ] && { echo "❌ ERROR: stress eval $METHOD/$SCENARIO failed"; exit $EXIT_CODE; }

    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
    echo "✓ Done stress: $METHOD / $SCENARIO"
    echo ""
}

# ==============================
# STAGE 1 — SMOKE TEST (60 ep)
# ==============================
echo "========================================="
echo "STAGE 1: Smoke test (60 episodes, all methods)"
echo "========================================="

LOG_DEMO="$OUTPUT_BASE/logs/demo_${SLURM_JOB_ID}.log"
python demo.py 2>&1 | tee "$LOG_DEMO"
cp -f demo_plots/*.png $OUTPUT_BASE/plots/ 2>/dev/null || true
echo "✓ Smoke test passed"
echo ""

# ==============================
# STAGE 2 — FULL TRAINING
# Paper Section 2.5: 5 000 episodes, normal scenario, all 3 methods
# ==============================
echo "========================================="
echo "STAGE 2: Full training — normal scenario (5 000 episodes)"
echo "========================================="

for METHOD in sa flat hma; do
    run_train $METHOD normal 5000
done

# ==============================
# STAGE 3 — STRESS EVALUATION
# Paper Section 3.4: evaluate HMA on stress scenarios
# using the pre-trained normal model
# ==============================
echo "========================================="
echo "STAGE 3: Stress evaluation — HMA-DRL only"
echo "========================================="

for SCENARIO in crit_load pv_outage dynamic_price high_res; do
    run_stress hma $SCENARIO
done

# ==============================
# STAGE 4 — COLLECT OUTPUTS
# ==============================
echo "========================================="
echo "STAGE 4: Collecting outputs"
echo "========================================="

cp -f plots/*.png         $OUTPUT_BASE/plots/       2>/dev/null || true
cp -f checkpoints/*.npz   $OUTPUT_BASE/checkpoints/ 2>/dev/null || true

echo "✓ All plots       → $OUTPUT_BASE/plots/"
echo "✓ All checkpoints → $OUTPUT_BASE/checkpoints/"

echo ""
echo "========================================="
echo "✅ ALL STAGES FINISHED"
echo "Finished: $(date)"
echo "========================================="
echo ""
echo "Results summary:"
echo "  Logs        → $OUTPUT_BASE/logs/"
echo "  Plots       → $OUTPUT_BASE/plots/"
echo "  Checkpoints → $OUTPUT_BASE/checkpoints/"
