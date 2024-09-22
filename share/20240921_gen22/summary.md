
# Simulation

Skipped. Reusing some previous datasets. We will bootstrap this new network.

## Collected

### Tsume

* training: 567840
* testing: 81180

# Train

## Trial 1: from the tsumes

### Play 1

#### baseline
```
Doctor's winrate: 57.00%
Plague's steps to win: 26.02 +/- 12.32
Doctor's steps to win: 32.88 +/- 14.60
```

#### 1.24 (20, 8) vs. random
```
Doctor's winrate: 45.00%
Plague's steps to win: 25.47 +/- 11.51
Doctor's steps to win: 30.84 +/- 11.41
```

#### gen21/3.24 (20, 8) vs. random
```
Doctor's winrate: 54.00%
Plague's steps to win: 24.39 +/- 12.26
Doctor's steps to win: 36.07 +/- 13.46
```

## Trial 2: smaller network

from 7x100 to 5x90.

### Play 2

#### baseline
```
Doctor's winrate: 57.00%
Plague's steps to win: 26.95 +/- 13.17
Doctor's steps to win: 31.33 +/- 13.75
```

#### Trial 2 (60) vs. random
```
Doctor's winrate: 55.00%
Plague's steps to win: 23.93 +/- 11.88
Doctor's steps to win: 31.53 +/- 11.43
```

#### Trial 2 (3) vs. random
```
Doctor's winrate: 61.00%
Plague's steps to win: 22.85 +/- 12.26
Doctor's steps to win: 30.13 +/- 13.69
```

### Play 3

Just some experimental runs (what are not?) to check the exclusiveness of GPU.
Well, that doesn't exist! So try how well will it go with pure GPU simulation runs.


