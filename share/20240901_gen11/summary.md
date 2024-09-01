
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
