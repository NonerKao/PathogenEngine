#/bin/bash

BATCH=200
BATCH_RATIO=5
PE_ROOT=/home/alankao/PathogenEngine/examples
SIM_ROOT=$(dirname $0)
SETUP_ROOT=$SIM_ROOT/setups-play2

function sim_p2d()
{
	THIS_ROOT=$SIM_ROOT/$(hostname)_$(date +%Y%m%d%H%M%S)
	mkdir -p $THIS_ROOT
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load-dir $SETUP_ROOT \
		--save-dir $THIS_ROOT \
		--batch $BATCH &
	sleep 3
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Plague -m $1 -b $BATCH --trial-unit $2 --delay-unit $3 &
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Doctor -m $4 -b $BATCH --trial-unit $5 --delay-unit $6 &
	
	wait
	find $THIS_ROOT -type f -size 0 -name '*.log' -delete
}

function sim_p2r()
{
	THIS_ROOT=$SIM_ROOT/$(hostname)_$(date +%Y%m%d%H%M%S)
	mkdir -p $THIS_ROOT
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load-dir $SETUP_ROOT \
		--save-dir $THIS_ROOT \
		--batch $BATCH &
	sleep 3
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Plague -m $1 -b $BATCH --trial-unit $2 --delay-unit $3 &
	python $PE_ROOT/coord_clients/main.py -t Query -s Doctor -b $BATCH &
	
	wait
	find $THIS_ROOT -type f -size 0 -name '*.log' -delete
}

function baseline()
{
	THIS_ROOT=$SIM_ROOT/$(hostname)_$(date +%Y%m%d%H%M%S)
	mkdir -p $THIS_ROOT
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load-dir $SETUP_ROOT \
		--save-dir $THIS_ROOT \
		--seed "$(date +%M%S)" \
		--batch $BATCH &
	sleep 3
	nc -zv 127.0.0.1 6241 & nc -zv 127.0.0.1 3698
	wait
	find $THIS_ROOT -type f -size 0 -name '*.log' -delete
	python $THIS_ROOT/../../simulation/stat.py $THIS_ROOT
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

if [ $1 == 1 ]; then
	echo "Baseline"
	baseline
	echo "10 simulation, No delay; Trial 1.5 vs. random"
	sim_p2r /mnt/20240914_gen19/train/game-trial1.pth.24 10 0
elif [ $1 == 2 ]; then
	echo "No simulation, No delay; Trial 2 vs. Trial 1.5"
	sim_p2d /mnt/20240914_gen19/train/game-trial2.pth.24 10 0 /mnt/20240914_gen19/train/game-trial1.pth.5 10 0
elif [ $1 == 3 ]; then
	echo "10 simulation, No delay; Trial 1.5 vs. Trial 2"
	sim_p2d /mnt/20240914_gen19/train/game-trial1.pth.5 10 0 /mnt/20240914_gen19/train/game-trial2.pth.24 10 0
fi
