#!/bin/bash -i

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate modAL
python3 -m experiments.torch_topics_ll_ideal
