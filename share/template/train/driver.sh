#!/bin/bash

TRIAL_ROOT=/mnt/20240818_gen4

pushd examples/coord_clients
python reinforcement_trainer.py -d $TRIAL_ROOT/simulation/general.bin \
	-m $TRIAL_ROOT/train/general.pth -n $TRIAL_ROOT/train/runs/gen4-general-0
popd
