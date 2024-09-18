
# Simulation

* gen19/train/plague-trial2.pth.24
  * (TRIAL, DELAY) = (15, 0), (16, 0), (14, 0)
  * setups-sim: 300/3 new for each ... interrupted
* gen19/train/plague-trial2.pth.24
  * (TRIAL, DELAY) = (18, 7)x3
  * setups-sim2: 500/2 new for each
* gen19/train/plague-trial2.pth.24
  * (TRIAL, DELAY) = (16, 4)x3
  * setups-sim3: 600/3 new for each

## Collected

* 461196 entries

# Train

## Trial 1

* Derived from gen16/train/trial2

## Trial 2

* Derived from gen19/train/trial2

## Trial 3: repeat Trial 2

This time, the 2/3 don't reproduce the miiricle gen19/trial2 showed.
Also, Trial 1 looks better.

Interesting, I want to see how it will be from scratch.

## Trial 4: from scratch

... it is extremely possible that the record for gen19/trial2 was wrong. It was from scratch instead of from some tsume-bootstraped ones?
Everything is so weird.

### Play 1

100 games from 1 sgf.
Compare trial 4.9 and trial 3.24.

#### baseline
```
Doctor's winrate: 46.00%
Plague's steps to win: 24.74 +/- 12.08
Doctor's steps to win: 32.43 +/- 11.62
```

#### 4.9 vs. random
```
Doctor's winrate: 38.00%
Plague's steps to win: 21.74 +/- 10.37
Doctor's steps to win: 33.26 +/- 10.96
```

#### 3.24 vs. random
```
Doctor's winrate: 39.00%
Plague's steps to win: 23.59 +/- 12.83
Doctor's steps to win: 30.26 +/- 10.55
```

#### 3.24 vs. 4.9
```
Doctor's winrate: 36.00%
Plague's steps to win: 20.12 +/- 10.55
Doctor's steps to win: 31.22 +/- 14.22
```

#### 4.9 vs. 3.24
```
Doctor's winrate: 42.00%
Plague's steps to win: 22.48 +/- 10.28
Doctor's steps to win: 29.86 +/- 9.56
```

### Play 2

100 games from 1 sgf.
Compare trial 4.24 and gen16/trial 2.24.

#### baseline
```
Doctor's winrate: 51.00%
Plague's steps to win: 26.18 +/- 11.07
Doctor's steps to win: 34.59 +/- 11.44
```

#### 4.24 vs. gen16/trial 2.24
```
Doctor's winrate: 49.00%
Plague's steps to win: 24.80 +/- 11.17
Doctor's steps to win: 31.55 +/- 13.27
```
#### gen16/trial 2.24 vs. 4.24
```
Doctor's winrate: 46.00%
Plague's steps to win: 26.04 +/- 12.40
Doctor's steps to win: 29.74 +/- 11.93
```

### Play 3

100 games from 1 sgf.
Compare trial 4.24 and 4.9.

#### baseline
```
Doctor's winrate: 64.00%
Plague's steps to win: 26.67 +/- 13.12
Doctor's steps to win: 31.34 +/- 12.64
```

#### 4.24 vs. 4.9
```
Doctor's winrate: 56.00%
Plague's steps to win: 25.77 +/- 14.10
Doctor's steps to win: 28.64 +/- 12.40
```

#### 4.9 vs. 4.24
```
Doctor's winrate: 55.00%
Plague's steps to win: 27.27 +/- 11.04
Doctor's steps to win: 31.64 +/- 12.29
```

## Trial 5: from 3.24

I want to know what will happen if I apply a set of tsume on this model again. Only for training, and keep using the same eval set.
Sounds fun!!!

* Training set: 169020 (gen16 extra_tsume.eval 57000 + gen17~gen20 110000)

### Play 4

#### baseline
```
Doctor's winrate: 53.00%
Plague's steps to win: 22.45 +/- 10.85
Doctor's steps to win: 34.11 +/- 12.71
```

#### 5.24 vs. 3.24
```
Doctor's winrate: 56.00%
Plague's steps to win: 27.05 +/- 11.81
Doctor's steps to win: 27.46 +/- 11.73
```

#### 5.24 vs. 5.24
```
Doctor's winrate: 61.00%
Plague's steps to win: 21.72 +/- 11.66
Doctor's steps to win: 28.75 +/- 11.38
```

## Trial 6: from scratch, datasets using trial 5 settings

... pass.
