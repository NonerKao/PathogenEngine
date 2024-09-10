
# Simulation

Skipped. Mainly take the tsume set and previous datasets for the training.

# Train

## Trial 1: new

* Training: 36000 (tsume) + 36000 (randomly selected from gen12 plague.train.bin)
* Testing: 12000 (tsume) + 12000 (randomly selected from gen12 plague.eval.bin)

* network: 8 x 108
* weight of heads: (0.5, 0, 0.5, 0)

Well, I don't reason much out of the results... Mixing seems to be a bad idea.

## Trial 2: new

* Training: 180000 (from gen12 plague.train.bin)
* Testing: 12000 (gen13 new_tsume.eval.bin)

* network: 8 x 108
* weight of heads: (0.25, 0.25, 0.25, 0.25)


## Trial 3: new

* Training: 180000 (from gen13 new_tsume.train.bin)
* Testing: 12000 (from gen12 plague.eval.bin)

* network: 8 x 108
* weight of heads: (0.25, 0.25, 0.25, 0.25)


## Trial 4: new, diff to trail 1
* weight of heads: (0.25, 0.25, 0.25, 0.25)

As expected in trial 2 and 3, both sets demonstrate quite different results. The way policy (distribution) are collected in normal simulation is not in control, and not sure how reliable they can turn out to be. When we train on one but test on the other, we can see they reach minimum loss quite fast.

An interesting result is from the comparison between trial 4 vs. 1. After the weight is adjusted, the knowledge the model gains is likely more general and moderate. We will see. I will let them fight against random and see what will happen.

# Play

## All trials, no further reading

Random (baseline): 55.67 %
gen13/plague-trial1.pth.11: 48.00 % ... Wow!!!
gen13/plague-trial2.pth.3: 52.33 %
gen13/plague-trial3.pth.4: 46.67 %
gen13/plague-trial4.pth.18: 54.00 % ... !!!

## All trials, further reading 10
gen13/plague-trial1.pth.11: 55.67 % ... Why???
gen13/plague-trial2.pth.3: 56.00 %
gen13/plague-trial3.pth.4: 51.67 %
gen13/plague-trial4.pth.18: ...... not finished.

## Continue some other experiment with gen13/plague-trial3.pth.*
Random (baseline) @c7db31f40a89_20240908134018 : 53.67%
gen13/plague-trial3.pth.4 @c7db31f40a89_20240908134342: 50.00%
gen13/plague-trial3.pth.60: 56.67%
gen13/plague-trial1.pth.11: 55.00%
gen13/plague-trial1.pth.60: 49.33%
