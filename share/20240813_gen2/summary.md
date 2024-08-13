# Summary

Play: Negate the value head. Otherwise, why are the models performing worse than random?

Well, it turns out as bad as negated.

# Simulation

Same as 20240812_gen1.

# Train

* TRAINING_BATCH_UNIT = 200
* TRAINING_OUTER_EPOCH = 20
* KFOLD = 5
* TRAINING_INNER_EPOCH = 5
* **ALPHA... = 0.1, 0.5, 0.3, 0.1**
* LR = 0.0004
* Network
* Others

# Play

## Setups

A new set of 100 sgfs, and each are played 10 times.

## Result

> in Doctor's win rate.

> Random in 6533 seed.

| Doctor\Plague |   Random    | Best | **General** | **Current** |
| ------------- | ----------- | ---- | ----------- | ----------- |
| Random        | 39.4%       | N/A  |             | 78.9%       |
| Best          | N/A         | N/A  | N/A         | N/A         |
| **General**   |             | N/A  |             |        |
| **Current**   | 20.4%       | N/A  |             |        |

## Statistics

### Random-Random
```
Doctor's winrate: 39.40%
Plague's steps to win: 30.83 +/- 15.50
Doctor's steps to win: 35.79 +/- 14.02
```

### Random-Plague
```
Doctor's winrate: 78.98%
Plague's steps to win: 33.87 +/- 20.91
Doctor's steps to win: 41.14 +/- 16.45

Stay rate: 4.49132%
Bad policy rate: 4.05066%
Invalidity rate: 15.5608%
```

### Doctor-Random
```
Doctor's winrate: 20.40%
Plague's steps to win: 30.81 +/- 15.61
Doctor's steps to win: 41.56 +/- 15.90

Stay rate: 3.74525%
Bad policy rate: 3.40554%
Invalidity rate: 19.7894%
```

## Check the intermediate models

### Random-Plague
