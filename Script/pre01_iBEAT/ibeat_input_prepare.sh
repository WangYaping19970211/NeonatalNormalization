#!/bin/bash
set -euo pipefail

# =====================================
# Usage: bash /project/4290000.01/yapwan/Projects/NeonatalNormalization/Script/pre01_iBEAT/ibeat_input_prepare.sh <dataset_name>
# =====================================

# =====================================
# Input argument
# =====================================
DATASET_NAME=${1:?Usage: $0 <dataset_name>}

BASE_DIR=/project/4290000.01/yapwan/Projects/NeonatalNormalization

DATA_DIR=${BASE_DIR}/Data/${DATASET_NAME}
TSV=${BASE_DIR}/Info/${DATASET_NAME}_participants.tsv
OUT=${BASE_DIR}/Info/${DATASET_NAME}_subjects_with_age.tsv

echo "Dataset: $DATASET_NAME"
echo "Data dir: $DATA_DIR"
echo "Participants TSV: $TSV"

echo -e "subject\tage_months" > "$OUT"

mapfile -t SUB_LIST < <(
    find "${DATA_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'sub-*' -printf '%f\n' | sort
)

for SUB in "${SUB_LIST[@]}"; do
    T1="${DATA_DIR}/${SUB}/${SUB}_T1w.nii.gz"
    T2="${DATA_DIR}/${SUB}/${SUB}_T2w.nii.gz"

    [[ -f "$T1" && -f "$T2" ]] || continue

    AGE_MONTHS=$(awk -F'\t' -v id="$SUB" -v dsname="$DATASET_NAME" '
        NR==1 {
            for (i=1; i<=NF; i++) {
                if ($i=="dataset") ds=i
                if ($i=="participant_id") pid=i
                if ($i=="age_months") age=i
            }
            next
        }
        $ds==dsname && $pid==id {
            print $age
            exit
        }' "$TSV")

    [[ -n "$AGE_MONTHS" ]] || continue

    echo -e "${SUB}\t${AGE_MONTHS}" >> "$OUT"
done

echo "Saved subject list to $OUT"

# Copy liscense file
cp /project/4290000.01/yapwan/Projects/CerebellarNormalization_1002/pipeline/ibeat_data/License "${DATA_DIR}/License"