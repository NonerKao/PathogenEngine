#/bin/bash

BATCH=500
BATCH_RATIO=4
PE_ROOT=/home/alankao/PathogenEngine/examples
SIM_ROOT=$CURR_ROOT/simulation
SETUP_ROOT=$SIM_ROOT/setups-sim7
MODEL=$PREV_ROOT/train2/plague-trial3.pth.22

function sim_p2d()
{
	THIS_ROOT=$SIM_ROOT/$(hostname)_$(date +%Y%m%d%H%M%S)
	mkdir -p $THIS_ROOT
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load-dir $SETUP_ROOT \
		--save-dir $THIS_ROOT \
		--batch $BATCH &
	sleep 3
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Plague -m $1 -b $BATCH --dataset "$THIS_ROOT/dataset" --trial-unit $2 --delay-unit $3 &
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Doctor -m $4 -b $BATCH --dataset "$THIS_ROOT/dataset" --trial-unit $5 --delay-unit $6 &
	
	wait
	find $THIS_ROOT -type f -size 0 -name '*.log' -delete
}

function prepare_setup(){
	if [ -d $1 ]; then
            return
        fi
	mkdir -p $1
	pushd $PE_ROOT
	for _ in $(seq 1 $(($BATCH/$BATCH_RATIO))); do id=$(uuidgen);
		RUST_BACKTRACE=1 cargo run --release --example setup_generator -- \
			--mode sgf --seed "$(echo $id | sed -e 's/-//g')" --save "$1/$id.sgf";
	done
	popd
}

prepare_setup $SETUP_ROOT
sim_p2d $MODEL 14 16 /dev/null 0 200
