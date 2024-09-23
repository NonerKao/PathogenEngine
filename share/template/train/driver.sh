#!/bin/bash

pushd examples/coord_clients
python reinforcement_trainer.py -d $(dirname $0)/../simulation-trial$1/game.train.bin \
	-t $(dirname $0)/../simulation-trial$1/game.eval.bin \
	-m $(dirname $0)/game-trial$1.pth.4 -n $(dirname $0)/runs/trial$1
popd
