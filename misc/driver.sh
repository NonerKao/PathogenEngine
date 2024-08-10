#/bin/bash

# This is used in container

BATCH=1000
MNT=/home/alankao/share
PE_ROOT=/home/alankao/PathogenEngine/examples

THIS_ROOT=$MNT/$(hostname)_$(date --utc +%Y%m%d%H%M%S)
mkdir -p $THIS_ROOT

ITER=0;
for I in $(ls $MNT/setups | shuf); do
	echo $ITER;
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load $MNT/setups/$I \
		--save $THIS_ROOT/$I
	ITER=$(($ITER + 1));
	sleep 1
	if [ $ITER -ge $BATCH ]; then
		exit 0
	fi
done &
sleep 1
python $PE_ROOT/coord_clients/main.py -t Reinforcement -s Doctor -b $BATCH --dataset "$THIS_ROOT/dataset" &
python $PE_ROOT/coord_clients/main.py -t Reinforcement -s Plague -b $BATCH --dataset "$THIS_ROOT/dataset"

wait
find $THIS_ROOT -type f -size 0 -name '*.log' -delete

# Use this to check the dataset size easily
# SUM=0; for S in $(find -name '*Doctor*.log' -exec bash -c 'echo $(($(ls -l {} | awk '"'{print \$5}'"')/4096))' \;); do SUM=$(($S+$SUM)); done; echo $SUM
# SUM=0; for S in $(find -name '*Plague*.log' -exec bash -c 'echo $(($(ls -l {} | awk '"'{print \$5}'"')/4096))' \;); do SUM=$(($S+$SUM)); done; echo $SUM
