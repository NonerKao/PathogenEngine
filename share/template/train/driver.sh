#!/bin/bash

pushd examples/coord_clients
python reinforcement_trainer.py -d $CURR_ROOT/simulation2/plague.train.bin \
	-t $CURR_ROOT/simulation2/plague.eval.bin \
	-m $CURR_ROOT/train2/plague-$1.pth -n $CURR_ROOT/train2/runs/$1
popd
