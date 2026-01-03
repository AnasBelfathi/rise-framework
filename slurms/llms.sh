#!/bin/bash
#SBATCH --job-name=ft-llm-lora
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH --constraint=a100  # partition A100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=bvh@a100

# ── Données personnelles anonymisées ──
# Par défaut: pas d’email dans le script.
# Si tu veux activer les mails, passe un email à l’exécution:
#   sbatch --export=ALL,SLURM_MAIL_USER="prenom.nom@domaine.tld" this_script.sh
#SBATCH --mail-type=NONE

#SBATCH --array=0-23%24   # plafond large, indices hors plage sortent proprement

set -euo pipefail

# ───── Config ÉDITABLE ─────
# Datasets attendus dans ../processed-datasets/<task>/{prepared.parquet,label2id.json}
TASKS=(PubMed_20k_RCT csabstracts biorc DeepRhole legal-eval scotus-category scotus-steps scotus-rhetorical_function)

# Modèles LLM (decoder-only)
MODELS=(
  mistralai/Mistral-7B-v0.3
  Qwen/Qwen3-8B
  meta-llama/Meta-Llama-3-8B
  #tiiuae/falcon-7b
)

# Seeds → crée runs/<task>/<exp_name>/seed-<seed>/
SEEDS=(1)

# exp_name PAR MODÈLE — obligatoire
declare -A EXP_NAME_BY_MODEL=(
  ["mistralai/Mistral-7B-v0.3"]="Mistral-7B-v0.3"
  ["Qwen/Qwen3-8B"]="Qwen3-8B"
  ["meta-llama/Meta-Llama-3-8B"]="Meta-Llama-3-8B"
)

# Hyperparams communs
EPOCHS=5
BATCH_SIZE=32
MAX_LEN=128
REPORT_TO="tensorboard"   # "none" | "tensorboard" | "wandb"

# Environnement
CONDA_ENV=env_hard
ANACONDA_MOD=anaconda-py3/2024.06

# Miroir local HF utilisé par LLMs/scripts/train.py (root_path = $DSDIR/HuggingFace_Models)
export DSDIR=${DSDIR:-/lustre/fsmisc/dataset}

# ───── Préparation env ─────
mkdir -p job_out_err
module purge
module load arch/a100
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

# Indices hors plage → sortie propre
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

# Chemins données (on est dans LLMs/)
DATA_PATH="../processed-datasets/${TASK}/prepared.parquet"
LABEL_MAP="../processed-datasets/${TASK}/label2id.json"
if [[ ! -f "$DATA_PATH" || ! -f "$LABEL_MAP" ]]; then
  echo "⛔ Données absentes pour task='$TASK':"
  echo "   $DATA_PATH"
  echo "   $LABEL_MAP"
  exit 1
fi

echo "▶ combo #$SLURM_ARRAY_TASK_ID/$TOTAL  →  task=$TASK | model=$MODEL_NAME | seed=$SEED | exp=$EXP_NAME"

# ───── TRAIN (LLMs) — script renommé ─────
# NB: ce script doit être lancé depuis le dossier LLMs/ (llms.py = LLMs/scripts/llms.py)
echo "=== TRAIN START ==="
srun python scripts/llms.py \
  --task_name "$TASK" \
  --exp_name "$EXP_NAME" \
  --model_name "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --label_map "$LABEL_MAP" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --seed "$SEED" \
  --report_to "$REPORT_TO" \
  --use_lora --infer \
  --load_in_4bit
echo "=== TRAIN DONE ==="

echo "✅ Terminé: $TASK | $MODEL_NAME | seed-$SEED | exp=$EXP_NAME"
