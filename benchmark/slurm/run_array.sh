#!/bin/bash
# SLURM array job for BlueBEAR: task N runs line N of the jobs file from make_jobs.py.
# Submit from the repository root (on RDS):
#
#   sbatch --array=1-$(wc -l < jobs.txt)%100 benchmark/slurm/run_array.sh jobs.txt
#
# Each task trains one (scenario, algorithm, seed) on one CPU. Runs checkpoint as they
# go, so a task that hits the time limit resumes when make_jobs.py lists it again.
# The log directory must exist before submitting (SLURM does not create it):
#   mkdir -p /rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs

#SBATCH --job-name=gridhunt
#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=0-10:00:00  # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs/%A_%a.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs/%A_%a.err

set -e

JOBS_FILE="${1:?usage: sbatch --array=1-N benchmark/slurm/run_array.sh jobs.txt}"
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")
if [ -z "${COMMAND}" ]; then
    echo "No line ${SLURM_ARRAY_TASK_ID} in ${JOBS_FILE}"
    exit 0
fi

module purge; module load bluebear

module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/gridhunt"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"

# Activate the environment (created once with benchmark/slurm/setup_env.sh)
mamba activate "${CONDA_ENV_PATH}"

# One thread per library: parallelism comes from the array, not inside a task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# No display on compute nodes (pygame is imported but never opens a window)
export SDL_VIDEODRIVER=dummy

cd "${SLURM_SUBMIT_DIR}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${COMMAND}"
${COMMAND} --threads "${SLURM_CPUS_PER_TASK:-1}"
echo "[$(date)] done"
