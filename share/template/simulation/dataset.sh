#!/bin/bash

SIZE=50000

time find ./ -name '*Doctor*.log' -exec cat {} >> doctor.raw.bin \;
time find ./ -name '*Plague*.log' -exec cat {} >> plague.raw.bin \;

python dataset_splitter.py plague.raw.bin $SIZE plague.bin plague.eval.bin 
python dataset_splitter.py doctor.raw.bin $SIZE doctor.bin doctor.eval.bin 

cat plague.bin >> general.raw.bin
cat doctor.bin >> general.raw.bin
python dataset_shuffler.py general.raw.bin general.bin $(($SIZE*2))
