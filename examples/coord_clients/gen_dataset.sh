#!/bin/bash

DIR=$1
if [ $# -lt 1 ]; then
	echo "Usage: $0 <DIR>"
	exit 0
fi
if [ ! -d $DIR ]; then
	mkdir $DIR
fi

ITER=2
for i in $(seq 1 $ITER); 
do 
	seed=$(date +%N)
	cargo run --example setup_generator -- --seed $seed -s temp.sgf
	cargo run --release --example coord_server -- -l temp.sgf &
	python main.py -v -r doc_temp.bin -t Random -s Doctor --seed $seed >doc.log &
	python main.py -v -r pla_temp.bin -t Random -s Plague --seed $seed >pla.log
	mkdir -p $DIR; echo $seed >> $DIR/seeds
	cat doc_temp.bin >> $DIR/doc.bin
	cat pla_temp.bin >> $DIR/pla.bin
done
