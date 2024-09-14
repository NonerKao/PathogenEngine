
# Simulation

* gen16/train/plague-trial2.pth.24
  * (TRIAL, DELAY) = [(10, 0), (10, 5), (10, 10)] x 3, interrupted
  * setups-sim: 500/4 new for each

## collected

* game (including both plague and doctor): 316193

# Train

There are some different strategies for this new step.

## Trial 1: continue gen16/train/plague-trial2.pth.24 with all game entries

* Training: 270000
* Testing: 42000

Not so sure about this, because the collected policy and values may still not be accurate enough.
But won't hurt to give it a try.

Interesting. 3 interesting points: 6 (`total` minimum), 16 (a late local minimun in saturated `value`), 24 (final one).

## Trial 2: continue gen16/train/plague-trial2.pth.24 with mix dataset

* Training: 135000 (new tsume) + 135000 (game)
* Testing: 42000 (same as Trial 1)

Moment like this, I would get myself into analysis paralyse.

It is weird that `test/valid` cannot match as well as trial one. In terms of valid, there should be no difference bettwen these two settings. Other than the growth of `test/value`, others `test/*` are all worse than trial one. Why?

Ocillasions can be seen in `val/*` sets, which is fine. It is also weird that other than `val/value`, others are better fitting than Trial 1's. Why?

### Play 1

Let's try these two, as plague, against the original and random.
#### Baseline
```
Doctor's winrate: 55.00%
Plague's steps to win: 19.22 +/- 4.09
Doctor's steps to win: 32.00 +/- 12.17
```

#### gen16/trial2.24
```
Doctor's winrate: 48.00%
Plague's steps to win: 23.62 +/- 11.11
Doctor's steps to win: 31.50 +/- 11.50
```

#### trial2.24
```
Doctor's winrate: 55.00%
Plague's steps to win: 28.51 +/- 14.61
Doctor's steps to win: 28.33 +/- 8.68
```

#### trial1.24
```
Doctor's winrate: 48.00%
Plague's steps to win: 24.19 +/- 10.95
Doctor's steps to win: 29.46 +/- 11.93
```

#### trial2.4
```
Doctor's winrate: 54.00%
Plague's steps to win: 23.48 +/- 11.70
Doctor's steps to win: 26.89 +/- 9.11
```

#### trial1.3
```
Doctor's winrate: 58.00%
Plague's steps to win: 24.90 +/- 11.49
Doctor's steps to win: 30.86 +/- 11.41
```

### Play 2: versus

#### trial1.24 (p) vs. gen16/trial2.24
```
Doctor's winrate: 53.00%
Plague's steps to win: 23.77 +/- 11.16
Doctor's steps to win: 31.55 +/- 12.49
```

#### gen16/trial2.24 (p) vs. trial1.24
```
Doctor's winrate: 47.00%
Plague's steps to win: 25.45 +/- 14.13
Doctor's steps to win: 32.09 +/- 12.22
```

## Trial 3: continue gen16/train/plague-trial2.pth.24 with all tsume entries

* Training: 180000 (extra tsume)
* Testing: 42000

Interrupted. Pretty the same as Trial 2
