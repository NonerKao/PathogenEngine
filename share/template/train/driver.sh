#!/bin/bash

pushd examples/coord_clients
python reinforcement_trainer.py -d $(dirname $0)/../simulation/game.train.bin \
	-t $(dirname $0)/../simulation/game.eval.bin \
	-m $(dirname $0)/game-$1.pth -n $(dirname $0)/runs/$1
popd
