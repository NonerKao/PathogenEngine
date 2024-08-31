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

Let it run, but seems still converged around 5~7. The combined eval set may not be a good idea. 

## Trial 2 (diff to Trial 1)

```
TRAINING_BATCH_UNIT = 60
```

Looks bad.


## Trial 3 (diff to Trial 1)

```
KFOLD = 8
TRAINING_OUTER_EPOCH = 1 // we won't go that far anyway
```

Looks bad.


## Trial 4 (diff to Trial 1)

```
KFOLD = 16
LEARNING_RATE = 0.0002
TRAINING_OUTER_EPOCH = 1 // we won't go that far anyway
```

Looks... good?

## Trial 5 (diff to Trial 1)
```
TRAINING_BATCH_UNIT = 20
KFOLD = 20
LEARNING_RATE = 0.0002
TRAINING_OUTER_EPOCH = 1 // we won't go that far anyway
```

Looks bad.
## Trial 6 (diff to Trial 1)
```
TRAINING_BATCH_UNIT = 10
KFOLD = 20
LEARNING_RATE = 0.0002
TRAINING_OUTER_EPOCH = 1 // we won't go that far anyway
```

Looks bad.

## Trial 7 (diff to Trial 6)

Network structure change to thin and tall:
```
RES_SIZE = 18
RES_INPUT_SIZE = 96
```



## Trial 8 (diff to trial 7)
```
TRAINING_BATCH_UNIT = 30
```

These two new network structure looks pretty bad. Their training loss seems large, and val loss is higher than old arch. Abandoned.

## Trial 9 (diff to trial 1)


```
RES_SIZE = 18
RES_INPUT_SIZE = 180
```
```
LEARNING_RATE = 0.0002
```

Shit, this was starting from new one...

## Trial 10 (diff to trial 9)

```
TRAINING_BATCH_UNIT = 20
KFOLD = 3
```
