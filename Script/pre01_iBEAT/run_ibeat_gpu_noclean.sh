#!/bin/bash
#SBATCH -J ib_a
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH -D /project/4290000.01/yapwan/Projects/NeonatalNormalization
#SBATCH -o logs/ibeat_%A_%a.out
#SBATCH -e logs/ibeat_%A_%a.err

set -euo pipefail

DATASET_NAME=${1:?Usage: run_ibeat_gpu.sh <dataset_name> <subjects.tsv>}
LIST=${2:?Usage: run_ibeat_gpu.sh <dataset_name> <subjects.tsv>}

module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

BASE_DIR=/project/4290000.01/yapwan/Projects/CerebellarNormalization_1002/pipeline/ibeat_suitpy
DATA_DIR=${BASE_DIR}/${DATASET_NAME}
SIF=/project/4290000.01/yapwan/toolbox/ibeat_release210.sif

# ---------- get subject & age ----------
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST" || true)
[[ -n "${LINE}" ]] || { echo "ERROR: empty LINE for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

SUB=$(echo "$LINE" | cut -f1)
AGE=$(echo "$LINE" | cut -f2)

echo "Dataset: ${DATASET_NAME}"
echo ">>> Array task ${SLURM_ARRAY_TASK_ID}: ${SUB} (age=${AGE})"

SUB_DIR="${DATA_DIR}/${SUB}"

echo ">>> ${SUB}: Proceeding to run iBEAT."

# Safety checks
[[ -d "$SUB_DIR" ]] || { echo "ERROR: Subject dir not found: $SUB_DIR"; exit 1; }
[[ -f "${SUB_DIR}/${SUB}_T1w.nii.gz" ]] || { echo "ERROR: Original T1w missing"; exit 1; }
[[ -f "${SUB_DIR}/${SUB}_T2w.nii.gz" ]] || { echo "ERROR: Original T2w missing"; exit 1; }

# ---------- license ----------
# if [[ ! -f "${DATA_DIR}/License" ]]; then
#   cp /project/4290000.01/yapwan/Projects/CerebellarNormalization_1002/pipeline/ibeat_data/License \
#      "${DATA_DIR}/License"
# fi
if [[ ! -f "${DATA_DIR}/License" ]]; then
  cp -n /project/4290000.01/yapwan/Projects/CerebellarNormalization_1002/pipeline/ibeat_data/License \
     "${DATA_DIR}/License" 2>/dev/null || true
fi

# ---------- pre-run checks ----------
[[ -f "${DATA_DIR}/License" ]] || { echo "ERROR: License file not found at ${DATA_DIR}/License"; exit 1; }
[[ -f "${DATA_DIR}/${SUB}/${SUB}_T1w.nii.gz" ]] || { echo "ERROR: Missing ${SUB}_T1w.nii.gz"; exit 1; }
[[ -f "${DATA_DIR}/${SUB}/${SUB}_T2w.nii.gz" ]] || { echo "ERROR: Missing ${SUB}_T2w.nii.gz"; exit 1; }

# ---------- container paths ----------
T1_IN=/InfantData/${SUB}/${SUB}_T1w.nii.gz
T2_IN=/InfantData/${SUB}/${SUB}_T2w.nii.gz
OUT_DIR=/InfantData

# ---------- tmp/cache ----------
export APPTAINER_TMPDIR=${DATA_DIR}/tmp
export APPTAINER_CACHEDIR=${DATA_DIR}/tmp
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---------- run ----------
echo ">>> Launching iBEAT for ${SUB}"
srun apptainer run --nv --containall --cleanenv --writable-tmpfs \
  -B ${DATA_DIR}:/InfantData \
  ${SIF} \
  --t1 ${T1_IN} \
  --t2 ${T2_IN} \
  --age ${AGE} \
  --out_dir ${OUT_DIR} \
  --sub_name ${SUB}