#!/bin/bash

function play_single()
{
	REC=records/d-$D_CONFIG-p-$P_CONFIG
        SETUP_DIR=./setups
	NUM_SETUP=$(ls $SETUP_DIR | wc -l)
	NUM_BATCH=$(($NUM_SETUP * $1))

	mkdir -p $REC
	for i in $(seq 1 $1); do
		find ./setups -name '*.sgf' -exec bash -c 'cargo run --example coord_server --release -- --load {} --save '"$REC"'/$(basename {})'".$i"' && sleep 1' \; ;
	done &
	if [ $2 == "random" ]; then
		python ../../../examples/coord_clients/main.py -t Query -s Plague -b $NUM_BATCH --seed $4
	else
		python ../../../examples/coord_clients/main.py -t ReinforcementPlay -s Plague -m ../train/$2.pth -b $NUM_BATCH | ./stat.awk | tee > $REC/p.log
	fi &

	if [ $3 == "random" ]; then
		python ../../../examples/coord_clients/main.py -t Query -s Doctor -b $NUM_BATCH --seed $4
	else
		python ../../../examples/coord_clients/main.py -t ReinforcementPlay -s Doctor -m ../train/$3.pth -b $NUM_BATCH | ./stat.awk | tee > $REC/d.log
	fi &

	wait

	echo $REC
	python ./stat.py $REC
	echo "==="
}

SEED=6533
NUM_REPEAT=10
P_CONFIG=random
D_CONFIG=doctor
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

P_CONFIG=plague
D_CONFIG=random
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

P_CONFIG=random
D_CONFIG=general
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

P_CONFIG=general
D_CONFIG=random
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG $SEED

P_CONFIG=plague
D_CONFIG=doctor
play_single $NUM_REPEAT $P_CONFIG $D_CONFIG
