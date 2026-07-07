#!/bin/bash
#SBATCH -J bme1309
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -D /project/4290000.01/yapwan/Projects/NeonatalNormalization
#SBATCH -o logs/bme_%j.out
#SBATCH -e logs/bme_%j.err

set -euo pipefail

module load apptainer

export APPTAINER_TMPDIR=/project/4290000.01/yapwan/toolbox/apptainer_tmp
export APPTAINER_CACHEDIR=/project/4290000.01/yapwan/toolbox/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

DATASET_NAME=HCPD
SUB=sub-1309

BIDS_DIR=/project/4290000.01/yapwan/Projects/NeonatalNormalization/Data/${DATASET_NAME}
OUTPUT_DIR=/project/4290000.01/yapwan/Projects/NeonatalNormalization/Data/${DATASET_NAME}
SIF=/project/4290000.01/yapwan/toolbox/bme-x_v1.0.2.sif

SUB_DIR=${BIDS_DIR}/${SUB}
ANAT_DIR=${SUB_DIR}/anat

# ---------- copy T1w and T2w into anat/ ----------
mkdir -p "${ANAT_DIR}"

for SUFFIX in T1w T2w; do
    SRC="${SUB_DIR}/${SUB}_${SUFFIX}.nii.gz"
    DST="${ANAT_DIR}/${SUB}_${SUFFIX}.nii.gz"

    if [[ -f "${SRC}" ]]; then
        cp "${SRC}" "${DST}"
        echo ">>> Copied ${SUB}_${SUFFIX}.nii.gz to anat/"
    else
        echo "WARNING: ${SRC} not found, skipping ${SUFFIX}"
    fi
done

# ---------- run BME-X for each suffix ----------
mkdir -p "${OUTPUT_DIR}"

for SUFFIX in T1w T2w; do

    [[ -f "${ANAT_DIR}/${SUB}_${SUFFIX}.nii.gz" ]] \
      || { echo "WARNING: ${SUFFIX} not in anat/, skipping..."; continue; }

    echo ">>> Launching BME-X for ${SUB} suffix=${SUFFIX}"

    srun apptainer exec --nv --pwd / \
      -B ${BIDS_DIR}:/bids_dir \
      -B ${OUTPUT_DIR}:/output_dir \
      ${SIF} \
      python3 /BME_X.py \
      --bids_dir   /bids_dir \
      --output_dir /output_dir \
      --subject    ${SUB} \
      --session    "" \
      --suffix     ${SUFFIX}

    echo ">>> Done: ${SUFFIX}"
done
