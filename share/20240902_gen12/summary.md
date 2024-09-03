
# Simulation

* gen11/train2/plague-trial2.pth (gen11 trial1 results)
  * (TRIAL, DELAY) = (6, 19), (7, 19), (7, 18), (8, 18)
  * 500 games each, from 125 setups
* gen11/train/plague-trial3.pth.22 <== shit, this was not intended
  * (TRIAL, DELAY) = (8, 19), (9, 17), (10, 16), (12, 15)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial5.pth.60
  * (TRIAL, DELAY) = (10, 18), (10, 17), (10, 16), (10, 15)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial4.pth.54
  * (TRIAL, DELAY) = (10, 7), (10, 8)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial3.pth.22
  * (TRIAL, DELAY) = (10, 19), (10, 17), (10, 16), (10, 15)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial3.pth.22
  * (TRIAL, DELAY) = (13, 13), (13, 12), (13,11), (13, 10), (13, 9)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial4.pth.54
  * (TRIAL, DELAY) = (10, 7), (10, 8)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial4.pth.54
  * (TRIAL, DELAY) = (10, 9), (10, 10), (20, 14), (20, 15)
  * 500 games each, from 125 new setups
* gen11/train2/plague-trial3.pth.22
  * (TRIAL, DELAY) = (14, 16)
  * 500 games each, from 125 new setups

# Train

 ## trial1: from gen11/train2/plague-trial4.pth.54
 
12x144

 ```
TRAINING_BATCH_UNIT = 100
TRAINING_INNER_EPOCH = 3
TRAINING_OUTER_EPOCH = 10

LEARNING_RATE = 0.0005
KFOLD = 10

ALPHA = 0.15
BETA = 0.7
GAMMA = 0.15
 ```
 
I may take this plague-trial1.pth.47.

## trial2: from gen11/train/plague-trial5.pth.60
 
12x288

 diff to trial1
 ```
  TRAINING_OUTER_EPOCH = 6
  LEARNING_RATE = 0.0007
 ```
 
I may take this plague-trial2.pth.65.

## trial3: train a new one, shall we?

* 16x128
* de-weight policy with the validity (understanding)
* Add the fourth term, misunderstanding back
