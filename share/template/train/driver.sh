#!/bin/bash

TRIAL_ROOT=/mnt/20240822_gen6

pushd examples/coord_clients
python reinforcement_trainer.py -d $TRIAL_ROOT/simulation/plague.train.bin \
	-t $TRIAL_ROOT/simulation/plague.eval.bin \
	-m $TRIAL_ROOT/train/plague-trial2.pth -n $TRIAL_ROOT/train/runs/gen6-trial2
popd
