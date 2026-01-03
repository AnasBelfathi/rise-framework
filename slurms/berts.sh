#!/bin/bash
#SBATCH --job-name=ssc-grid
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
# Option 1 (recommandée) : ne pas mettre d’email dans le script.
# Option 2 : si tu veux activer les mails, passe un email à sbatch via:
#   sbatch --export=ALL,SLURM_MAIL_USER="prenom.nom@domaine.tld" this_script.sh
#SBATCH --mail-type=NONE

# Plafond large; les indices hors plage sortent proprement (pas besoin d’argument à sbatch)
#SBATCH --array=0-39%40

set -euo pipefail

# ───── Config ÉDITABLE ─────
# Datasets attendus dans processed-datasets/<task>/{prepared.parquet,label2id.json}
TASKS=(PubMed_20k_RCT csabstracts biorc DeepRhole legal-eval scotus-category scotus-steps scotus-rhetorical_function)

# ⚠️ Modèles BERT uniquement (compatibles avec ton script d’entraînement).
MODELS=(
  bert-base-uncased
  roberta-base
  deberta-base
  albert-base-v2
  distilbert-base-uncased
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
)

# Hyperparams communs
EPOCHS=5
BATCH_SIZE=32
MAX_LEN=128

# Environnement
CONDA_ENV=env_hard
ANACONDA_MOD=anaconda-py3/2024.06

# Miroir local HF utilisé par les scripts via $DSDIR/HuggingFace_Models
export DSDIR=${DSDIR:-/lustre/fsmisc/dataset}

# ───── Préparation env ─────
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

# ── Activation optionnelle des emails via variable d’environnement (anonymisée par défaut) ──
# Exemple:
#   sbatch --export=ALL,SLURM_MAIL_USER="x@y.z" script.sh
#   (et dans ce cas, tu peux aussi remplacer mail-type=NONE par mail-type=ALL si voulu)
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
  echo "📧 Notifications SLURM configurées via SLURM_MAIL_USER (valeur fournie à l’exécution)."
fi

echo "=== Node: $(hostname) ==="
nvidia-smi || true
python -V

# ───── Construction des combos (task, model, seed) ─────
COMBOS=()
for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      COMBOS+=("$task|$model|$seed")
    done
  done
done
TOTAL=${#COMBOS[@]}

# Indices hors plage → sortie propre (permet le plafond large de --array)
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "Index ${SLURM_ARRAY_TASK_ID} > TOTAL ${TOTAL} → rien à faire."
  exit 0
fi

IFS='|' read -r TASK MODEL_NAME SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

# exp_name défini PAR MODÈLE (exigé)
if [[ -z "${EXP_NAME_BY_MODEL[$MODEL_NAME]+x}" ]]; then
  echo "⛔ Aucun exp_name défini pour MODEL='$MODEL_NAME' dans EXP_NAME_BY_MODEL"
  exit 2
fi
EXP_NAME="${EXP_NAME_BY_MODEL[$MODEL_NAME]}"

# Chemins données
DATA_PATH="../processed-datasets/${TASK}/prepared.parquet"
LABEL_MAP="../processed-datasets/${TASK}/label2id.json"
if [[ ! -f "$DATA_PATH" || ! -f "$LABEL_MAP" ]]; then
  echo "⛔ Données absentes pour task='$TASK':"
  echo "   $DATA_PATH"
  echo "   $LABEL_MAP"
  exit 1
fi

echo "▶ combo #$SLURM_ARRAY_TASK_ID/$TOTAL  →  task=$TASK | model=$MODEL_NAME | seed=$SEED | exp=$EXP_NAME"

# ───── TRAIN (script renommé) ─────
echo "=== TRAIN START ==="
srun python scripts/berts.py \
  --task_name "$TASK" \
  --exp_name "$EXP_NAME" \
  --model_name "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --label_map "$LABEL_MAP" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --seed "$SEED"
echo "=== TRAIN DONE ==="
