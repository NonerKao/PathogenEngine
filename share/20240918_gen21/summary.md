
# Simulation

* gen20/train/game-trial3.pth.24 self-play
  * (TRIAL, DELAY) = (20, 0), (21, 0), (22, 0)
  * setups-sim: 500/5 new for each
* gen20/train/game-trial3.pth.24 vs. random
  * (TRIAL, DELAY) = (10, 0) x 3
  * setups-sim: 500/5 new for each
* gen20/train/game-trial3.pth.24 self-play
  * (TRIAL, DELAY) = (10, 0) x 3 x 2
  * setups-sim: 500/5 new for each

## Collected

* 636106 entries

# Train

## Trial 1: derived from gen20/game-trial3.pth.24

### Play 1

* 100/20 sgfs

#### baseline
```
Doctor's winrate: 54.00%
Plague's steps to win: 25.96 +/- 13.18
Doctor's steps to win: 30.81 +/- 10.27
```

#### 1.8 vs. 1.20
```
Doctor's winrate: 51.00%
Plague's steps to win: 22.14 +/- 10.82
Doctor's steps to win: 29.02 +/- 10.68
```

#### 1.8 vs. random
```
Doctor's winrate: 40.00%
Plague's steps to win: 24.07 +/- 8.76
Doctor's steps to win: 28.85 +/- 11.88
```

#### 1.20 vs. 1.8
```
Doctor's winrate: 55.00%
Plague's steps to win: 25.18 +/- 15.08
Doctor's steps to win: 31.53 +/- 14.02
```

#### 1.20 vs. random
```
Doctor's winrate: 42.00%
Plague's steps to win: 21.45 +/- 10.59
Doctor's steps to win: 30.86 +/- 12.31
```

#### 1.8 selfplay
```
Doctor's winrate: 61.29%
Plague's steps to win: 21.33 +/- 11.18
Doctor's steps to win: 30.84 +/- 11.44
```

#### 1.20 selfplay
data missing. somehow the play was aborted.

## Trial 2: derived from gen20/game-trial5.pth.24

## Trial 3: from scratch

### Play 2

#### baseline
```
Doctor's winrate: 53.00%
Plague's steps to win: 27.89 +/- 12.66
Doctor's steps to win: 32.53 +/- 13.19
```

#### Trial 3.24 (10) vs. gen19/trial 2.24 (10)
```
Doctor's winrate: 56.00%
Plague's steps to win: 26.23 +/- 13.86
Doctor's steps to win: 30.32 +/- 11.02
```

#### Trial 3.24 (15) vs. gen19/trial 2.24 (5)
```
Doctor's winrate: 56.00%
Plague's steps to win: 25.45 +/- 13.02
Doctor's steps to win: 31.61 +/- 11.90
```

#### gen19/trial 2.24 (10) vs. Trial 3.24 (10)
```
Doctor's winrate: 53.00%
Plague's steps to win: 24.06 +/- 11.32
Doctor's steps to win: 28.91 +/- 11.84
```

#### gen19/trial 2.24 (15) vs. Trial 3.24 (5)
```
Doctor's winrate: 60.00%
Plague's steps to win: 26.60 +/- 17.48
Doctor's steps to win: 31.90 +/- 14.09
```

#### Trial 3.24 (10) selfplay
```
Doctor's winrate: 58.00%
Plague's steps to win: 27.67 +/- 10.84
Doctor's steps to win: 30.90 +/- 11.43
```

#### gen19/trial 2.24 (10) selfplay
```
Doctor's winrate: 52.00%
Plague's steps to win: 24.88 +/- 11.92
Doctor's steps to win: 29.85 +/- 9.68
```

#### Trial 3.24 (10) vs. random
```
Doctor's winrate: 50.00%
Plague's steps to win: 23.88 +/- 12.67
Doctor's steps to win: 30.80 +/- 10.38
```

#### gen19/trial 2.24 (10) vs. random
```
Doctor's winrate: 47.00%
Plague's steps to win: 25.98 +/- 14.60
Doctor's steps to win: 33.62 +/- 12.69
```
