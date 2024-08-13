#!/bin/bash

function play_single()
{
	NUM_REPEAT=20
	ROOT=/home/alankao/PathogenEngine
        MNT=/mnt/20240813_gen2
	PLAY_ROOT=$MNT/play
	TRAIN_ROOT=$MNT/train
	REC=$PLAY_ROOT/records/d-$D_CONFIG-p-$P_CONFIG
	NUM_SETUP=$(ls $PLAY_ROOT/setups | wc -l)
	NUM_BATCH=$(($NUM_SETUP * $NUM_REPEAT))

	mkdir -p $REC
	for i in $(seq 1 $1); do
		find $PLAY_ROOT/setups2 -name '*.sgf' -exec bash -c 'cargo run --example coord_server --release -- --load {} --save '"$REC"'/$(basename {})'".$i"' && sleep 1' \; ;
	done &
	if [ $2 == "random" ]; then
		python $ROOT/examples/coord_clients/main.py -t Query -s Plague -b $NUM_BATCH --seed $4
	else
		python $ROOT/examples/coord_clients/main.py -t ReinforcementPlay -s Plague -m $TRAIN_ROOT/$2.pth.$NUM_SETUP -b $NUM_BATCH | $PLAY_ROOT/stat.awk | tee $REC/p.log
	fi &

	if [ $3 == "random" ]; then
		python $ROOT/examples/coord_clients/main.py -t Query -s Doctor -b $NUM_BATCH --seed $4
	else
		python $ROOT/examples/coord_clients/main.py -t ReinforcementPlay -s Doctor -m $TRAIN_ROOT/$3.pth.$NUM_SETUP -b $NUM_BATCH | $PLAY_ROOT/stat.awk | tee $REC/d.log
	fi &

	wait

	echo $REC
	python $PLAY_ROOT/stat.py $REC
	echo "==="
}

SEED=6533
P_CONFIG=random
D_CONFIG=doctor
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED
