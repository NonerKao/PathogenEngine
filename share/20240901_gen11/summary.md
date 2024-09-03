
# Simulation

* gen10/plague-trial2.pth (gen9 results)
  * DELAY = 19, 18, 17, 16
  * TRIAL = 12
* gen10/plague-trial4.pth.5 (just a few sets. 642aa4a56a7e_20240831150344 looks good)
  * DELAY = 17, 16
  * TRIAL = 12

Nothing special about the winrate. Should we care?
```
Doctor's winrate: 49.90%
Plague's steps to win: 26.72 +/- 12.15
Doctor's steps to win: 32.69 +/- 11.86
```

# Train

Aggregating some previous used in gen10 ...
train, eval = (81000 + 27000 (gen9), 18000)

## Trial 1

```
TRAINING_BATCH_UNIT = 15
TRAINING_INNER_EPOCH = 2
TRAINING_OUTER_EPOCH = 3

LEARNING_RATE = 0.0003
KFOLD = 10

ALPHA = 0.33
BETA = 0.33
GAMMA = 0.33
```

Shit, the test set has been seen.

## Trial 2 (explore new architecture)

Parameters are the same as Trial 1, but widen each CNN layer, as 12x**288**.

Looks interesting.

# Simulation2

Mainly drived from gen10/plague-trial2.pth.10.
Because of its size, it seems collecting slow.

# Train2

Training set: 20000 (gen9) + 70000 (gen10) + 60000 (gen11 simulation1) + 30000 (gen11 simulation2)
Eval set: 3000 (gen10) + 3000 (gen11 simulation2)

## trial1: from gen11/train/plague-trial1.pth.3

```
 12 TRAINING_BATCH_UNIT = 100
 13 TRAINING_INNER_EPOCH = 1
 14 TRAINING_OUTER_EPOCH = 3
 15 
 16 LEARNING_RATE = 0.0005
 17 KFOLD = 10
 18 
 19 ALPHA = 0.1
 20 BETA = 0.45
 21 GAMMA = 0.45
```

## trial2: from gen11/train2/plague-trial1.pth.27

The drop near the end epoch is so intriguing.

## trial3: from gen11/train2/plague-trial2.pth.30

I will take plague-trial3.pth.22 for the simulation of the next generation.

## trial4: from gen11/train/plague-trial2.pth.10

Note that this has a different shape, also, hypermeter diff to trial1
```
 14 TRAINING_OUTER_EPOCH = 6
```

I was waiting for the drop!!! Cool, what is that!?

I will take plague-trial4.pth.54 for the simulation of the next generation.

## trial5: from gen11/train/plague-trial1.pth.3

diff to trial1
```
 TRAINING_OUTER_EPOCH = 6
 LEARNING_RATE = 0.0007
```

Looks good, as well? I want this plague-trial5.pth.60.

## trial6: from gen11/train/plague-trial1.pth.3
diff to trial1
```
 TRAINING_BATCH_UNIT = 200
 TRAINING_OUTER_EPOCH = 6
 LEARNING_RATE = 0.0008
```

Interrupted. Won't matter much.

## trial7: from gen11/train/plague-trial1.pth.3
diff to trial1
```
 TRAINING_BATCH_UNIT = 20
 TRAINING_OUTER_EPOCH = 2
 KFOLD = 20
 
 ALPHA = 0.1
 BETA = 0.7
 GAMMA = 0.2
```

Would like to interrupt. Doesn't make much difference.
