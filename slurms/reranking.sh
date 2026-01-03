#!/bin/bash
#SBATCH --job-name=hard-hybrid-table
#SBATCH --output=./job_out_err/%x_%A_%j.out
#SBATCH --error=./job_out_err/%x_%A_%j.err
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=bvh@v100

# ── Données personnelles anonymisées ──
#SBATCH --mail-type=NONE

set -euo pipefail

# ───── Config ÉDITABLE ─────
CONDA_ENV=env_hard
ANACONDA_MOD=anaconda-py3/2024.06

# Chemin vers ton script Python (adapte si besoin)
PY_SCRIPT="scripts/reranking.py"

# (Optionnel) Dossier HF local si utilisé ailleurs dans ton repo
export DSDIR=${DSDIR:-/lustre/fsmisc/dataset}

# Paramètres par défaut
SEED="${SEED:-1}"
LATEX_OUT="${LATEX_OUT:-model_hard_hybrid_table.tex}"

# Mode 1: gather (recommandé pour générer le tableau global)
GATHER="${GATHER:-1}"   # 1 = --gather ; 0 = run sur (TASK,EXP)

# Mode 2: run simple (si GATHER=0)
TASK="${TASK:-scotus-rhetorical_function}"
EXP="${EXP:-roberta-base}"

# Embeddings (optionnels) : si vides, le script garde sa logique interne
EMB_EXP="${EMB_EXP:-}"      # ex: bge-m3
EMB_SEED="${EMB_SEED:-}"    # ex: 1

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

echo "=== Node: $(hostname) ==="
python -V

# ───── Exécution ─────
ARGS=(--seed "${SEED}")

if [[ "${GATHER}" == "1" ]]; then
  # Génère un seul tableau LaTeX en scannant outputs/*__*__seed-<seed>__test.jsonl
  ARGS+=(--task "${TASK}" --exp "${EXP}" --gather --latex-out "${LATEX_OUT}")
else
  # Évalue seulement (TASK,EXP) (ton script actuel n’imprime pas forcément tout en mode non-gather,
  # mais on garde l’appel standard)
  ARGS+=(--task "${TASK}" --exp "${EXP}" --latex-out "${LATEX_OUT}")
fi

if [[ -n "${EMB_EXP}" ]]; then
  ARGS+=(--emb-exp "${EMB_EXP}")
fi
if [[ -n "${EMB_SEED}" ]]; then
  ARGS+=(--emb-seed "${EMB_SEED}")
fi

echo "=== RUN: python ${PY_SCRIPT} ${ARGS[*]} ==="
srun python "${PY_SCRIPT}" "${ARGS[@]}"

echo "✅ Done. Output tex: ../${LATEX_OUT} (selon ROOT dans le script)"
