#!/bin/bash

DIR=/opt/dataset/random_dataset_1K
ITER=1000

for i in $(seq 1 $ITER); 
do 
	seed=$(date +%N)
	cargo run --example setup_generator -- --seed $seed -s temp.sgf
	cargo run --release --example coord_server -- -l temp.sgf &
	python main.py -v -r doc_temp.bin -t Random -s Doctor --seed $seed >/dev/null &
	python main.py -v -r pla_temp.bin -t Random -s Plague --seed $seed >/dev/null
	mkdir -p $DIR; echo $seed >> $DIR/seeds
	cat doc_temp.bin >> $DIR/doc.bin
	cat pla_temp.bin >> $DIR/pla.bin
done
