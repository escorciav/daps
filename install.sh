#!/bin/bash
# ```
#
# Sample bash-script used to setup our package
#
# This script just create the conda environment and all our
# python-dependencies. It will not install all the dependencies
# such as gcc, cuda.
#
# Usage: ./install.sh OR sh install.sh
# Requirements: conda
#
# ```
set -e
# Shortcut in case you wanna use another name for the environment
script_dirname=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
conda_env_name=daps-eccv16

# Create conda environment
# check conda existence
if hash conda 2>/dev/null; then
  # check if environment exists
  if [ ! "$(conda env list | grep "$conda_env_name")" ]; then
    # Use appropriate YAML-file, so far just x64
    conda env create -f $script_dirname/"environment_x64.yml"
  else
    source activate $conda_env_name
    conda env update -f $script_dirname/"environment_x64.yml"
  fi
else
  echo "Conda is not installed"
  return -1
fi

source activate $conda_env_name

# Install the same version of Lasagne/Theano used in our work
pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git@88648d8d5531deb4e5e4201a3663ffbc1465b84d
pip install --upgrade --no-deps git+https://github.com/Lasagne/Lasagne.git@9886da26df40cbde9222d4e20706b4b21bbdb627
