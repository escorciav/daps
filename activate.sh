# ```
#
# Sample bash-script used to setup enviroment variables
#
# Usage: . setenv.sh OR source setenv.sh
# Requirements: modules-environment
#
# ```

# Shortcut in case you wanna use another name for the environment
conda_env_name=daps-eccv16
# Get project dir
daps_dirname=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Fun starts from this line

# Expose dependencies to your shell
# Clean everything (added by modules)
module purge
# GCC (you may wanna try a new one)
module load compilers/gcc/4.9.4
# CUDA (new one? OK, try it out!)
module load compilers/cuda/7.5
# CUDNN (new one? of course, go ahead!)
module load libs/cudnn/v5/1-7.5
# Finally, add conda
module load tools/conda

# Activate conda enviroment
source activate $conda_env_name
# Push daps-dir in your PYTHONPATH (make our code visible)
export PYTHONPATH=$daps_dirname:$PYTHONPATH
