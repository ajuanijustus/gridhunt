#!/bin/bash
# One-off: create the conda environment used by run_array.sh. Submit from the
# repository root (on RDS):
#
#   mkdir -p /rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs
#   sbatch benchmark/slurm/setup_env.sh
#
# Installs only what the benchmark needs. Ray is not needed (only the human-experiment
# case study uses it). torch 2.5.1 is used because the pinned torch 2.3.0 cannot use
# NumPy 2 on Linux ("Numpy is not available"). Only pygame-ce is installed (plain
# pygame conflicts with it). gridhunt is installed with --no-deps so setup.py's pins
# cannot override these choices.

#SBATCH --job-name=gridhunt_setup
#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault
#SBATCH --time=0-01:00:00  # days-hours:minutes:seconds

#SBATCH --output=/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs/%x_%j.out
#SBATCH --error=/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs/%x_%j.err

set -e

module purge; module load bluebear

module load bear-apps/2022b
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

CONDA_ENV_PATH="/rds/projects/b/baberc-human-agent-teaming/Aju/${USER}_conda_envs/gridhunt"
export CONDA_PKGS_DIRS="/scratch/${USER}/conda_pkgs"

# Create environment with Python first
mamba create --yes --prefix "${CONDA_ENV_PATH}" python=3.12

# Activate environment
mamba activate "${CONDA_ENV_PATH}"

pip install --upgrade pip setuptools wheel

pip install \
  "numpy==2.4.4" \
  "gymnasium==0.28.1" \
  "pettingzoo==1.24.2" \
  "pygame-ce==2.5.5" \
  "pygame-gui==0.6.14" \
  "matplotlib==3.10.8" \
  "tqdm==4.67.1" \
  "pytest==7.4.0"

pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu

cd "${SLURM_SUBMIT_DIR}"
pip install -e . --no-deps

# Check the environment: versions, the NumPy-torch bridge, and the test suite
export SDL_VIDEODRIVER=dummy
python -c "import numpy, torch, pygame, gridhunt; print('numpy', numpy.__version__, '| torch', torch.__version__, '| pygame-ce', pygame.version.ver, '|', torch.from_numpy(numpy.zeros(2)))"
python -m pytest tests benchmark/tests -q -p no:cacheprovider

echo "GridHunt environment set-up successfully!"
