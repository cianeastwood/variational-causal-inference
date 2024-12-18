#!/bin/bash

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name morphoMNIST-test"
PYARGS="$PYARGS --data_name morphoMNIST"
PYARGS="$PYARGS --data_path $DATA/data/morphoMNIST"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --hparams hparams/hparams_morphoMNIST.json"
PYARGS="$PYARGS --gpu 0" #PYARGS="$PYARGS --cpu"

PYARGS="$PYARGS --omega0 10.0"
PYARGS="$PYARGS --omega1 0.02"
PYARGS="$PYARGS --omega2 0.01"
PYARGS="$PYARGS --dist_outcomes bernoulli"
PYARGS="$PYARGS --dist_mode discriminate"
#PYARGS="$PYARGS --checkpoint_classifier /path/to/trained/classifier"

PYARGS="$PYARGS --max_epochs 200"
PYARGS="$PYARGS --batch_size 256"
PYARGS="$PYARGS --checkpoint_freq 2"


# Wandb
PYARGS="$PYARGS --use_wandb"

python main.py $PYARGS
