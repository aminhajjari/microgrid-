#!/bin/bash
#SBATCH --job-name=hma_drl_microgrid
#SBATCH --account=def-arashmoh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/gkianfar/scratch/Amin/MSH/output/logs/hma_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/MSH/output/logs/hma_%j.err

# ==========================================================================
# CHANGES vs. original  (every change is marked FIX-*)
# ==========================================================================
#
# FIX-S  Stage 1 smoke test now exits the job with a non-zero code when
#        HMA reward < 50 % of SA reward.  Previously the job continued even
#        with a broken HMA (ratio was 0.38 in the failed run).
#
# FIX-K  Stage 2b seed loop no longer overwrites the checkpoint on every
#        seed.  train.py (FIX-1) handles best-checkpoint logic internally;
#        the shell loop just needs to run each seed and let Python decide.
#        We also preserve each seed's result .npz under a unique name
#        (hma_normal_seed{N}_results.npz) so all 10 seeds are available for
#        Fig 12 post-processing, instead of the original which overwrote the
#        same file 10 times.
#
# ==========================================================================

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

# FIX-S: demo.py now exits with code 1 if HMA reward < 50 % of SA reward.
# The 'set -e' at the top of this script will catch that and abort the job,
# saving the remaining GPU hours for a fixed re-submission.
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ STAGE 1 SMOKE TEST FAILED (exit code $EXIT_CODE)"
    echo "   HMA reward is too low relative to SA.  Fix hma_drl.py and resubmit."
    exit $EXIT_CODE
fi

cp -f demo_plots/*.png $OUTPUT_BASE/plots/ 2>/dev/null || true
echo ""

# ==============================
# STAGE 2 — FULL TRAINING
# ==============================
echo "========================================="
echo "STAGE 2: Full training — normal scenario (5 000 episodes)"
echo "========================================="

for METHOD in sa flat hma; do
    run_train $METHOD normal 5000
done

# ==============================
# STAGE 2b — 10 SEEDS FOR FIG 12
# ==============================
echo "========================================="
echo "STAGE 2b: HMA — 10 independent seeds (Fig 12)"
echo "========================================="

# FIX-K: each seed writes its results to a unique .npz so no seed overwrites
# another.  train.py (FIX-1) uses save_if_best() so hma_weights.pt always
# holds the best checkpoint seen across all seeds.
for SEED in 1 2 3 4 5 6 7 8 9 10; do
    LOG="$OUTPUT_BASE/logs/hma_seed${SEED}_${SLURM_JOB_ID}.log"
    echo "  → seed $SEED"

    python train.py \
        --method   hma     \
        --scenario normal  \
        --episodes 5000    \
        --batch    256     \
        --device   cuda    \
        2>&1 | tee "$LOG"

    # FIX-K: rename this seed's result file so it is not overwritten
    SRC="$OUTPUT_BASE/checkpoints/hma_normal_results.npz"
    DST="$OUTPUT_BASE/checkpoints/hma_normal_seed${SEED}_results.npz"
    if [ -f "$SRC" ]; then
        cp -f "$SRC" "$DST"
        echo "    Seed $SEED results → $DST"
    fi

    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done
echo "✓ All 10 seeds done"
echo ""

# ==============================
# STAGE 3 — STRESS EVALUATION
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
