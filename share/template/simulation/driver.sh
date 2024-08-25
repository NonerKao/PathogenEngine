#/bin/bash

BATCH=200
MNT=/mnt/20240824_gen7
PE_ROOT=/home/alankao/PathogenEngine/examples

function sim_single()
{
	THIS_ROOT=$MNT/simulation/$(hostname)_$(date --utc +%Y%m%d%H%M%S)
	mkdir -p $THIS_ROOT
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load-dir $THIS_ROOT/../setups \
		--save-dir $THIS_ROOT \
		--batch $BATCH &
	sleep 3
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Plague -m $1 -b $BATCH --dataset "$THIS_ROOT/dataset" --trial-unit $2 --delay-unit $3 &
	python $PE_ROOT/coord_clients/main.py -t ReinforcementSimulate -s Doctor -m $4 -b $BATCH --dataset "$THIS_ROOT/dataset" --trial-unit $5 --delay-unit $6
	
	wait
	find $THIS_ROOT -type f -size 0 -name '*.log' -delete
}

sim_single /mnt/20240824_gen7/simulation/empty_$(date --utc +%Y%m%d%H%M%S).pth 0 200 /mnt/20240824_gen7/simulation/empty_$(date --utc +%Y%m%d%H%M%S).pth 0 200

