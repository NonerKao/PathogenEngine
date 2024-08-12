
# Summary

There are two critical bugs that might have contributed to the poor result,
so basically this is an endpoint. We will not proceed from here.

# Simulation
* DELAY = 20
* TRIAL = 60
* Temparature = 2.0
* Others

# Train
* ALPHA ... = (0.05, 0.90, 0.05)
* LR = 0.0004
* Network
  * RES_SIZE = 12
  * RES_INPUT_SIZE = 84
  * NATURE_CHANNEL_SIZE = (8 + 2 + 1 + 2 + 11)

# Play

## Setups

A new set of 100 sgfs, and each are played 10 times.

## Result

> in Doctor's win rate.

> Random in 6533 seed.

| Doctor\Plague |   Random    | Best | **General** | **Current** |
| ------------- | ----------- | ---- | ----------- | ----------- |
| Random        | 39.7%       | N/A  | 62.0%       | 56.5%       |
| Best          | N/A         | N/A  | N/A         | N/A         |
| **General**   | 19.4%       | N/A  | 35.5%       | 32.9%       |
| **Current**   | 26.4%       | N/A  | 45.8%       | 37.7%       |

## Statistics

### General Self-Play
```
Stay rate: 4.29231%
Bad policy rate: 25.6482%
Invalidity rate: 7.59098%

Stay rate: 5.07099%
Bad policy rate: 22.8117%
Invalidity rate: 7.29261%
```

### General-General
```
Doctor's winrate: 35.50%
Plague's steps to win: 45.13 +/- 19.46
Doctor's steps to win: 50.46 +/- 16.38
```

### General-Plague
```
Doctor's winrate: 32.90%
Plague's steps to win: 41.98 +/- 18.55
Doctor's steps to win: 50.40 +/- 16.50
```

### Doctor-General
```
Doctor's winrate: 45.80%
Plague's steps to win: 42.52 +/- 19.71
Doctor's steps to win: 45.93 +/- 16.31
```

### Random-Random
```
Doctor's winrate: 39.67%
Plague's steps to win: 30.97 +/- 14.70
Doctor's steps to win: 36.24 +/- 15.00
```

### Doctor-Random
```
Doctor's winrate: 26.40%
Plague's steps to win: 32.27 +/- 15.73
Doctor's steps to win: 41.98 +/- 16.65
```

### Random-Plague
```
Doctor's winrate: 56.50%
Plague's steps to win: 37.39 +/- 16.30
Doctor's steps to win: 38.44 +/- 15.03
```

### General-Random
```
records/d-general-p-random
Doctor's winrate: 19.40%
Plague's steps to win: 34.43 +/- 16.83
Doctor's steps to win: 47.77 +/- 15.50
```

### Random-General
```
Doctor's winrate: 62.00%
Plague's steps to win: 38.64 +/- 17.39
Doctor's steps to win: 40.36 +/- 15.74
```

### Doctor-Plague
```
Doctor's winrate: 37.70%
Plague's steps to win: 40.71 +/- 18.53
Doctor's steps to win: 43.54 +/- 16.05
```
