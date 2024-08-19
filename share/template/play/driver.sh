#!/bin/bash

ROOT=/home/alankao/PathogenEngine
MNT=/mnt/20240818_gen4
PLAY_ROOT=$MNT/play
TRAIN_ROOT=$MNT/train
NUM_REPEAT=10

function play_single()
{
	REC=$PLAY_ROOT/records/$(echo d-$D_CONFIG-p-$P_CONFIG | sed -e 's/\//#/g')
	NUM_SETUP=$(ls $PLAY_ROOT/setups | wc -l)
	NUM_BATCH=$(($NUM_SETUP * $NUM_REPEAT))

	mkdir -p $REC
	for i in $(seq 1 $1); do
		find $PLAY_ROOT/setups -name '*.sgf' -exec bash -c 'cargo run --example coord_server --release -- --load {} --save '"$REC"'/$(basename {})'".$i"' && sleep 1' \; ;
	done &
	if [ $2 == "random" ]; then
		python $ROOT/examples/coord_clients/main.py -t Query -s Plague -b $NUM_BATCH --seed $4
	else
		python $ROOT/examples/coord_clients/main.py -t ReinforcementPlay -s Plague -m $2 -b $NUM_BATCH | $PLAY_ROOT/stat.awk | tee $REC/p.log
	fi &

	if [ $3 == "random" ]; then
		python $ROOT/examples/coord_clients/main.py -t Query -s Doctor -b $NUM_BATCH --seed $4
	else
		python $ROOT/examples/coord_clients/main.py -t ReinforcementPlay -s Doctor -m $3 -b $NUM_BATCH | $PLAY_ROOT/stat.awk | tee $REC/d.log
	fi &

	wait

	echo $REC
	python $PLAY_ROOT/stat.py $REC
	echo "==="
}

SEED=6533
P_CONFIG=random
D_CONFIG=random
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

for i in $(seq 1 23) $(seq 24 12 60) $(seq 72 24 120); do
	P_CONFIG=$TRAIN_ROOT/general.pth.$i
	D_CONFIG=random
	play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

	P_CONFIG=random
	D_CONFIG=$TRAIN_ROOT/general.pth.$i
	play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

	P_CONFIG=$TRAIN_ROOT/general.pth.$i
	D_CONFIG=$TRAIN_ROOT/general.pth.$i
	play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED
done
