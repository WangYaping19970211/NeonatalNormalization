#!/bin/bash
# usage:
# bash submit_ibeat.sh <dataset_name> <subjects.tsv> [noclean]
# bash /project/4290000.01/yapwan/Projects/NeonatalNormalization/Script/pre01_iBEAT/submit_ibeat.sh BCP /project/4290000.01/yapwan/Projects/NeonatalNormalization/Info/BCP_subjects_with_age_30_eachbin.tsv
# bash /project/4290000.01/yapwan/Projects/NeonatalNormalization/Script/pre01_iBEAT/submit_ibeat.sh dHCP /project/4290000.01/yapwan/Projects/NeonatalNormalization/Info/dHCP_subjects_with_age.tsv
# bash /project/4290000.01/yapwan/Projects/NeonatalNormalization/Script/pre01_iBEAT/submit_ibeat.sh BCP /project/4290000.01/yapwan/Projects/CerebellarNormalization_1002/pipeline/ibeat_suitpy/Info/BCP_subjects_with_age.tsv noclean

DATASET=${1:?Usage: submit_ibeat.sh <dataset_name> <subjects.tsv> [noclean]}
TSV=${2:?Usage: submit_ibeat.sh <dataset_name> <subjects.tsv> [noclean]}
MODE=${3:-""}

N=$(($(wc -l < "$TSV") - 1))

echo "Dataset: $DATASET"
echo "Input TSV: $TSV"
echo "Subjects: $N"

if [ "$N" -le 0 ]; then
    echo "No subjects found in $TSV"
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")

if [[ "$MODE" == "noclean" ]]; then
    RUN_SCRIPT="${SCRIPT_DIR}/run_ibeat_gpu_noclean.sh"
else
    RUN_SCRIPT="${SCRIPT_DIR}/run_ibeat_gpu.sh"
fi

echo "Using script: $RUN_SCRIPT"

sbatch --array=1-${N}%5 \
    "$RUN_SCRIPT" \
    "$DATASET" "$TSV"