#!/bin/bash
# Export the trained teams for gridhunt.zoo and check them (see benchmark/export_zoo.py).
# Submit from the repository root through submit.sh:
#
#   benchmark/slurm/submit.sh benchmark/slurm/export_zoo.sh benchmark/results/marl zoo_export

#SBATCH --job-name=gridhunt-zoo
#SBATCH --time=0-03:00:00  # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -e

RUNS="${1:?usage: submit.sh benchmark/slurm/export_zoo.sh RUNS_DIR OUT_DIR}"
OUT="${2:?usage: submit.sh benchmark/slurm/export_zoo.sh RUNS_DIR OUT_DIR}"

source benchmark/slurm/activate.sh
echo "[$(date)] exporting ${RUNS} -> ${OUT} on $(hostname)"
python benchmark/export_zoo.py --runs "${RUNS}" --out "${OUT}"
echo "[$(date)] done"
