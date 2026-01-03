#!/bin/bash
#SBATCH --job-name=ft-multiple-neg-all
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=bvh@v100

# ── Données personnelles anonymisées ──
# Par défaut: pas d’email dans le script.
# Si tu veux activer les mails, passe un email à l’exécution:
#   sbatch --export=ALL,SLURM_MAIL_USER="prenom.nom@domaine.tld" this_script.sh
#SBATCH --mail-type=NONE

# Plafond large; indices hors plage sortent proprement
#SBATCH --array=0-47%48

set -euo pipefail

# ───── Config ÉDITABLE ─────
# Datasets attendus dans processed-datasets/<task>/{prepared.parquet,label2id.json}
TASKS=(PubMed_20k_RCT csabstracts biorc DeepRhole legal-eval scotus-category scotus-steps scotus-rhetorical_function)

# ⚠️ Modèles (BERT-like + bge-m3) compatibles avec finetuning_multiple_neg.py
MODELS=(
  bert-base-uncased
  roberta-base
  deberta-base
  albert-base-v2
  distilbert-base-uncased
  BAAI/bge-m3
)

# Seeds → crée runs/<task>/<exp_name>/seed-<seed>/
SEEDS=(1)

# exp_name PAR MODÈLE —> À RENSEIGNER ICI
declare -A EXP_NAME_BY_MODEL=(
  ["bert-base-uncased"]="bert-base-uncased"
  ["roberta-base"]="roberta-base"
  ["deberta-base"]="deberta-base"
  ["albert-base-v2"]="albert-base-v2"
  ["distilbert-base-uncased"]="distilbert-base-uncased"
  ["BAAI/bge-m3"]="bge-m3"
)

# Hyperparams communs à tous les jobs
EPOCHS=10
BATCH_SIZE=32
MAX_SEQ_LEN=128
LR=2e-5
WARMUP=0.1
SCALE=20.0                  # MNRL temperature = 1/scale
NEG_PER_ANCHOR=3            # crée negative_1..negative_K
LABEL_TEMPLATE="LABEL: {label}"

# (Optionnel) Miroir HF local; laissé intact si déjà exporté
export DSDIR=${DSDIR:-/lustre/fsmisc/dataset}

# Environnement
CONDA_ENV=env_hard
ANACONDA_MOD=anaconda-py3/2024.06

# Script Python à lancer (renommé)
SCRIPT_PY="scripts/confusion_contrastive_learning.py"

# ───────── Préparation env ─────────
mkdir -p job_out_err
module purge
module load "${ANACONDA_MOD}"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ── Activation optionnelle des emails via variable d’environnement ──
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
  echo "📧 Notifications SLURM configurées via SLURM_MAIL_USER (valeur fournie à l’exécution)."
fi

echo "=== Node: $(hostname) ==="
nvidia-smi || true
python -V

# ───────── Construction des combos (task, model, seed) ─────────
COMBOS=()
for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      COMBOS+=("$task|$model|$seed")
    done
  done
done
TOTAL=${#COMBOS[@]}

# Indices hors plage → sortie propre
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "Index ${SLURM_ARRAY_TASK_ID} > TOTAL ${TOTAL} → rien à faire."
  exit 0
fi

IFS='|' read -r TASK_NAME MODEL_NAME SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

# exp_name par modèle (exigé)
if [[ -z "${EXP_NAME_BY_MODEL[$MODEL_NAME]+x}" ]]; then
  echo "⛔ Aucun exp_name défini pour MODEL='$MODEL_NAME' dans EXP_NAME_BY_MODEL"
  exit 2
fi
EXP_NAME="${EXP_NAME_BY_MODEL[$MODEL_NAME]}"

echo "▶ combo #$SLURM_ARRAY_TASK_ID/$TOTAL  →  task=$TASK_NAME | model=$MODEL_NAME | seed=$SEED | exp=$EXP_NAME"

# ───────── TRAIN ─────────
echo "=== TRAIN START ==="
srun python "${SCRIPT_PY}" \
  --task_name "${TASK_NAME}" \
  --exp_name "${EXP_NAME}" \
  --model_name "${MODEL_NAME}" \
  --label_template "${LABEL_TEMPLATE}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --warmup_ratio "${WARMUP}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --scale "${SCALE}" \
  --num_negatives_per_anchor "${NEG_PER_ANCHOR}" \
  --no_amp
echo "=== TRAIN DONE ==="

echo "✅ Terminé: $TASK_NAME | $MODEL_NAME | seed-$SEED | exp=$EXP_NAME"
