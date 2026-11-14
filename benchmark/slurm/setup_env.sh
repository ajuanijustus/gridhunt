#!/bin/bash
# One-off: create the conda environment the benchmark jobs use, with the exact versions that
# produced the published results (benchmark/requirements.txt), then run the test suites.
# Submit from the repository root through submit.sh:
#
#   benchmark/slurm/submit.sh benchmark/slurm/setup_env.sh

#SBATCH --job-name=gridhunt-setup
#SBATCH --time=0-01:00:00  # days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source benchmark/slurm/cluster.env
eval "${CONDA_SETUP}"

conda create --yes --prefix "${ENV_PREFIX}" python=3.12
conda activate "${ENV_PREFIX}"
pip install --upgrade pip
pip install -r benchmark/requirements.txt
# --no-deps keeps the pinned versions above
pip install -e . --no-deps

export SDL_VIDEODRIVER=dummy
python -c "import numpy, torch, gridhunt; print('numpy', numpy.__version__, '| torch', torch.__version__, '| gridhunt', gridhunt.__version__)"
python -m pytest tests benchmark/tests -q -p no:cacheprovider
echo "Environment ready: ${ENV_PREFIX}"
