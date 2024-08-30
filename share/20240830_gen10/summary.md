Directly derived from gen9.

# Simulation

* DELAY = 18, 17, 16
* TRIAL = 10

* take gen9 plague.pth.5~7

Nothing special about the winrate. Should we care?
```
Doctor's winrate: 51.50%
Plague's steps to win: 25.55 +/- 12.72
Doctor's steps to win: 31.62 +/- 12.67
```

# Train

## Trail 1

train, eval = (96000, 3600 + 3600)

```
TRAINING_BATCH_UNIT = 15
TRAINING_INNER_EPOCH = 2
TRAINING_OUTER_EPOCH = 3

LEARNING_RATE = 0.0001
KFOLD = 10

ALPHA = 0.33
BETA = 0.33
GAMMA = 0.33
```

* Results
