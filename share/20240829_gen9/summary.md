
# Simulation

* DELAY = 20
* TRIAL = 100

```
Doctor's winrate: 51.61%
Plague's steps to win: 25.48 +/- 12.71
Doctor's steps to win: 31.90 +/- 12.29
```

Hmm... It doesn't look so different.

# Train

## Trail 1
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

* Change loss of valid head from `sum` to `mean`
