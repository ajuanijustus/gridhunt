# Sourced by the job scripts: load this cluster's settings and activate the benchmark environment.
set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source benchmark/slurm/cluster.env
eval "${CONDA_SETUP}"
conda activate "${ENV_PREFIX}"
# One thread per library: parallelism comes from the array, not inside a task
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# No display on compute nodes
export SDL_VIDEODRIVER=dummy
# Use this checkout's gridhunt even if another copy is installed in the environment
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
