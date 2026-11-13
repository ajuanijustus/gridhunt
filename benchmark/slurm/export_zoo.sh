#!/bin/bash
# Export the benchmark's trained teams for gridhunt.zoo and check them (see benchmark/export_zoo.py).
# Submit from the repository root (on RDS), pointing at the finished runs:
#
#   sbatch benchmark/slurm/export_zoo.sh benchmark/results/marl zoo_export
#
# The log directory must exist before submitting (SLURM does not create it):
#   mkdir -p /rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs

#SBATCH --job-name=gridhunt-zoo
#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=0-03:00:00  # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs/zoo_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs/zoo_%j.err

set -e

RUNS="${1:?usage: sbatch benchmark/slurm/export_zoo.sh RUNS_DIR OUT_DIR}"
OUT="${2:?usage: sbatch benchmark/slurm/export_zoo.sh RUNS_DIR OUT_DIR}"

module purge; module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/gridhunt"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"
mamba activate "${CONDA_ENV_PATH}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy

cd "${SLURM_SUBMIT_DIR}"
# Use this checkout's gridhunt even if another copy is installed in the environment
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "[$(date)] exporting ${RUNS} -> ${OUT} on $(hostname)"
python benchmark/export_zoo.py --runs "${RUNS}" --out "${OUT}"
echo "[$(date)] done"
