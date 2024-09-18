
# Simulation

* gen16/train/plague-trial2.pth.24
  * (TRIAL, DELAY) = (12, 0) x 2 (per container) x 3 (#containers)
  * setups-sim: 400/4 new for each

## Collected

* 144502 (gen18 sim) + 316193 (gen17 sim)

# Train

## Trial 1

Interesting to compare to gen17/trial1. It is not clear why gen17/trial1 cannot retrieve common knowledge from the game data. test/policy and test/value goes up immediately.

The differences are (in this trial 1)

* Larger dataset (training 402660 vs. 270000, testing: 57540 vs. 42000)
* Newly collected game play data are simulated with higher trial count (12 vs. 10)
* New dataset processing: randomly shuffled before splitted

Training/testing with game data purely seems to saturate at some point. This makes sense because we don't think the game data are 100% correct but tsume data are.

Totally speaking, I would like to know how will 11 (a local minimum at `policy`) and 24 (a final run) perform against previous one and random

### play

trail1.24 vs. random
```
Doctor's winrate: 42.00%
Plague's steps to win: 23.69 +/- 11.49
Doctor's steps to win: 32.05 +/- 14.15
```

trail1.24 vs. gen16/trial2.24
```
Doctor's winrate: 44.00%
Plague's steps to win: 24.68 +/- 14.89
Doctor's steps to win: 30.27 +/- 11.00
```

trail1.11 vs. random
```
Doctor's winrate: 49.00%
Plague's steps to win: 22.29 +/- 11.70
Doctor's steps to win: 29.18 +/- 10.43
```

trail1.11 vs. gen16/trial2.24
```
Doctor's winrate: 45.00%
Plague's steps to win: 23.47 +/- 12.75
Doctor's steps to win: 29.20 +/- 11.48
```
