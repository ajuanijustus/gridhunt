#!/bin/bash
# SLURM array job: task N runs line N of the jobs file from make_jobs.py. Submit from the
# repository root through submit.sh, which adds the cluster's account and log settings:
#
#   benchmark/slurm/submit.sh --time=1-12:00:00 --array=1-$(wc -l < jobs.txt)%100 benchmark/slurm/run_array.sh jobs.txt
#
# Each task trains one (scenario, algorithm, seed) on one CPU. Runs checkpoint as they go, so a
# task that hits the time limit resumes when make_jobs.py lists it again.

#SBATCH --job-name=gridhunt
#SBATCH --time=0-10:00:00  # days-hours:minutes:seconds; the full benchmark used 1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -e

JOBS_FILE="${1:?usage: submit.sh --array=1-N benchmark/slurm/run_array.sh jobs.txt}"
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")
if [ -z "${COMMAND}" ]; then
    echo "No line ${SLURM_ARRAY_TASK_ID} in ${JOBS_FILE}"
    exit 0
fi

source benchmark/slurm/activate.sh
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${COMMAND}"
${COMMAND} --threads "${SLURM_CPUS_PER_TASK:-1}"
echo "[$(date)] done"
