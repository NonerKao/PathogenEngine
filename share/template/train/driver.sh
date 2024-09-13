#!/bin/bash

pushd examples/coord_clients
python reinforcement_trainer.py -d $(dirname $0)/../simulation/tsume/extra_tsume.train.bin \
	-t $(dirname $0)/../simulation/tsume/extra_tsume.eval.bin \
	-m $(dirname $0)/plague-$1.pth -n $(dirname $0)/runs/$1
popd
