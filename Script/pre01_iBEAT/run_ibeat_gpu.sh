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
# LIST=${BASE_DIR}/Info/${DATASET_NAME}_subjects_with_age.tsv

module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

BASE_DIR=/project/4290000.01/yapwan/Projects/NeonatalNormalization

DATA_DIR=${BASE_DIR}/Data/${DATASET_NAME}

SIF=/project/4290000.01/yapwan/toolbox/ibeat_release210.sif

# ---------- get subject & age ----------
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST" || true)
[[ -n "${LINE}" ]] || { echo "ERROR: empty LINE for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

SUB=$(echo "$LINE" | cut -f1)
AGE=$(echo "$LINE" | cut -f2)

echo "Dataset: ${DATASET_NAME}"
echo ">>> Array task ${SLURM_ARRAY_TASK_ID}: ${SUB} (age=${AGE})"

SS_T1="${DATA_DIR}/${SUB}/T1-skullstripped.nii.gz"
SS_T2="${DATA_DIR}/${SUB}/T2-skullstripped.nii.gz"

SUB_DIR="${DATA_DIR}/${SUB}"

# ---------- CASE 1: skull-stripped exit → quit ----------
if [[ -f "$SS_T1" && -f "$SS_T2" ]]; then
  echo ">>> ${SUB}: skull-stripped outputs already exist. Skip iBEAT."
  exit 0
fi

# ---------- CASE 2: skull-stripped not exit → clean intermediate files ----------
echo ">>> ${SUB}: skull-stripped outputs missing. Cleaning intermediate files."

# Safety checks
[[ -d "$SUB_DIR" ]] || { echo "ERROR: Subject dir not found: $SUB_DIR"; exit 1; }
[[ -f "${SUB_DIR}/${SUB}_T1w.nii.gz" ]] || { echo "ERROR: Original T1w missing"; exit 1; }
[[ -f "${SUB_DIR}/${SUB}_T2w.nii.gz" ]] || { echo "ERROR: Original T2w missing"; exit 1; }

# Delete everything except original T1w / T2w
find "$SUB_DIR" -mindepth 1 -maxdepth 1 \
  ! -name "${SUB}_T1w.nii.gz" \
  ! -name "${SUB}_T2w.nii.gz" \
  -exec rm -rf {} +

echo ">>> ${SUB}: intermediate files cleaned. Continue to run iBEAT."

# ---------- license ----------
if [[ ! -f "${DATA_DIR}/License" ]]; then
  cp /project/4290000.01/yapwan/Projects/CerebellarNormalization_1002/pipeline/ibeat_data/License \
     "${DATA_DIR}/License"
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

# ---------- helpers ----------
is_file_stable () {
  local f="$1"
  local wait_sec="${2:-30}"  
  [[ -f "$f" ]] || return 1
  local s1 s2
  s1=$(stat -c %s "$f" 2>/dev/null || echo 0)
  sleep "$wait_sec"
  s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [[ "$s1" -eq "$s2" && "$s1" -gt 0 ]]
}

SUB_DIR="${DATA_DIR}/${SUB}"
KEEP_FILES=(
  "${SUB}_T1w.nii.gz"
  "${SUB}_T2w.nii.gz"
  "T1-skullstripped.nii.gz"
  "T2-skullstripped.nii.gz"
)
cleanup () {
  echo "[ $(date '+%Y-%m-%d %H:%M:%S') ] cleanup for ${SUB}"

  if [[ "$SKULLSTRIP_SUCCESS" != true ]]; then
    echo ">>> Skull-strip not confirmed successful; skip cleanup."
    return
  fi

  for f in "${SUB_DIR}"/*; do
    [[ -e "$f" ]] || continue
    fname=$(basename "$f")

    keep=false
    for k in "${KEEP_FILES[@]}"; do
      [[ "$fname" == "$k" ]] && keep=true && break
    done

    if [[ "$keep" == false ]]; then
      rm -rf "$f"
    fi
  done
}
trap cleanup EXIT

# ---------- run (background for early-stop) ----------
echo ">>> Launching iBEAT for ${SUB}"
srun apptainer run --nv --containall --cleanenv --writable-tmpfs \
  -B ${DATA_DIR}:/InfantData \
  ${SIF} \
  --t1 ${T1_IN} \
  --t2 ${T2_IN} \
  --age ${AGE} \
  --out_dir ${OUT_DIR} \
  --sub_name ${SUB} &
APPTAINER_PID=$!
echo ">>> Apptainer PID=${APPTAINER_PID}"

# ---------- early-stop monitor ----------
SKULLSTRIP_SUCCESS=false
while kill -0 "${APPTAINER_PID}" 2>/dev/null; do
  if [[ -f "${SS_T1}" && -f "${SS_T2}" ]] \
     && is_file_stable "${SS_T1}" 30 \
     && is_file_stable "${SS_T2}" 30; then
    echo "[ $(date '+%Y-%m-%d %H:%M:%S') ] Skull-stripped outputs detected & stable for ${SUB}."
    SKULLSTRIP_SUCCESS=true
    echo "[ $(date '+%Y-%m-%d %H:%M:%S') ] Early-stopping iBEAT (kill -TERM ${APPTAINER_PID})."
    kill -TERM "${APPTAINER_PID}" 2>/dev/null || true
    sleep 5
    break
  fi
  sleep 30
done

wait "${APPTAINER_PID}" 2>/dev/null || true

# ---------- final check ----------
if [[ -f "$SS_T1" && -f "$SS_T2" ]]; then
  echo ">>> ${SUB} finished (skull-strip present)."
  exit 0
else
  echo "ERROR: ${SUB} finished but skull-stripped outputs missing."
  exit 2
fi