#/bin/bash

BATCH=1000
TSUME_SET=20
BATCH_RATIO=1
PE_ROOT=/home/alankao/Documents/MetaSelf/SideProjects/PathogenEngine/examples
SIM_ROOT=$PE_ROOT/../share/20240910_gen15/simulation/tsume
SETUP_ROOT=$SIM_ROOT/setups

function sim_r2r()
{
	THIS_ROOT=$SIM_ROOT/$(date +%Y%m%d%H%M%S)
	mkdir -p $THIS_ROOT
	RUST_BACKTRACE=1 cargo run --release --example coord_server -- \
		--load-dir $SETUP_ROOT \
		--save-dir $THIS_ROOT \
		--batch $BATCH &
	sleep 3
	nc -vz 127.0.0.1 3698 & nc -vz 127.0.0.1 6241
	
	wait
	sleep 3
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

prepare_setup $SETUP_ROOT 2>/dev/null
for i in $(seq 1 $TSUME_SET); do
	sim_r2r
done
cargo run --release --example tsume_generator -- -l $SIM_ROOT --dataset extra_tsume.raw.bin
