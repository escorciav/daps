# ```
#
# Sample bash-script used to clean enviroment variables
#
# Usage: . setenv.sh OR source setenv.sh
# Requirements: modules-environment
#
# ```

# Deactivate current conda environment
source deactivate
# Clean your PYTHONPATH (Warning: it's a complete reset)
export PYTHONPATH=
# Clean everything loaded by modules-environment
module purge
