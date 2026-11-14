#!/bin/bash
# Submit a benchmark SLURM script with this cluster's settings (benchmark/slurm/cluster.env).
# Run from the repository root; any sbatch options go before the script:
#
#   benchmark/slurm/submit.sh benchmark/slurm/setup_env.sh
#   benchmark/slurm/submit.sh --time=1-12:00:00 --array=1-720%100 benchmark/slurm/run_array.sh jobs.txt
#   benchmark/slurm/submit.sh benchmark/slurm/export_zoo.sh benchmark/results/marl zoo_export

set -e
CONFIG="benchmark/slurm/cluster.env"
if [ ! -f "${CONFIG}" ]; then
    echo "Missing ${CONFIG}: copy benchmark/slurm/cluster.env.example and fill it in (run from the repository root)." >&2
    exit 1
fi
source "${CONFIG}"
mkdir -p "${LOG_DIR}"

# Array tasks get one log per task (job id _ task id), other jobs one log per job id
PATTERN="%x_%j"
for arg in "$@"; do
    case "${arg}" in --array*|-a) PATTERN="%x_%A_%a" ;; esac
done
OPTIONS=(--account="${SLURM_ACCOUNT}" --output="${LOG_DIR}/${PATTERN}.out" --error="${LOG_DIR}/${PATTERN}.err")
if [ -n "${SLURM_QOS}" ]; then
    OPTIONS+=(--qos="${SLURM_QOS}")
fi
sbatch "${OPTIONS[@]}" "$@"
