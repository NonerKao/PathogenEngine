
# Simulation

* gen18/train/plague-trial1.pth.24
  * (TRIAL, DELAY) = (15, 0) x 1 (per container) x 3 (#containers)
  * setups-sim: 200/4 new for each
* gen18/train/plague-trial1.pth.24
  * (TRIAL, DELAY) = (15, 5) x 1 (per container) x 3 (#containers)
  * setups-sim: 800/2 new for each
* gen18/train/plague-trial1.pth.24
  * (TRIAL, DELAY) = (15, 5) x 1 (per container) x 3 (#containers)
  * setups-sim: 800/2 new for each

## Collected

* 457619 entries

# Train

## Trial 1: from gen18/trial1

Nothing too weird. `test/policy` goes up higher than gen18/trial1 itself.

## Trial 2: from gen16/trial2

Was originally for a reference but I didn't expect that it looks better than gen18/trial1. I don't know how to explain it.
Key questoin: train/policy worse than trial 1, but test/policy better? train/value and test value on par.

### Play 1

#### Baseline
```
Doctor's winrate: 50.00%
Plague's steps to win: 23.68 +/- 10.63
Doctor's steps to win: 31.60 +/- 12.57
```
~~Well, this is interesting enough. The standard deviation is less than previous records. We happend to have randomized a set of setups that are more friendly to Plague.~~ It was due to a bug in `coord_server` setting for pure random game plays.


#### No simulation, No delay; Trial 1 vs. Trial 2
```
Doctor's winrate: 56.00%
Plague's steps to win: 26.98 +/- 14.31
Doctor's steps to win: 32.68 +/- 12.30
```
#### No simulation, No delay; Trial 2 vs. Trial 1
```
Doctor's winrate: 56.50%
Plague's steps to win: 25.55 +/- 12.21
Doctor's steps to win: 32.76 +/- 13.69
```
It is hard to explain the two no-sim scenarios. Not clear why the setting is friendly to Doctor.

#### 10 simulation, No delay; Trial 1 vs. Random
```
Doctor's winrate: 55.50%
Plague's steps to win: 22.84 +/- 10.14
Doctor's steps to win: 30.70 +/- 12.91
```
This is unbelivablely bad. Cannot imagine why this can happen.

#### 10 simulation, No delay; Trial 2 vs. Random
```
Doctor's winrate: 40.00%
Plague's steps to win: 24.58 +/- 11.57
Doctor's steps to win: 28.57 +/- 9.25
```
What can we tell from the results? Maybe a model shouldn't live too long?
Recall: trained from tsumes (gen16/trial2), simulated and trained (gen18/trial1), simulated and trained **but derived from gen/trial2**.

#### 10 simulation, No delay; Trial 1 vs. Trial 2
```
Doctor's winrate: 51.27%
Plague's steps to win: 26.15 +/- 12.34
Doctor's steps to win: 30.08 +/- 11.45
```

#### 10 simulation, No delay; Trial 2 vs. Trial 1
```
Doctor's winrate: 46.00%
Plague's steps to win: 24.00 +/- 12.27
Doctor's steps to win: 30.72 +/- 10.70
```

### Play 2

It is really hard for me to believe there is nothing trial1 can do in its live time. Take the earlier one (trial1.5).

#### Baseline
```
Doctor's winrate: 54.50%
Plague's steps to win: 23.40 +/- 11.24
Doctor's steps to win: 31.38 +/- 12.71
```

#### 10 simulation, No delay; Trial 1.24 (shit, didn't change this) vs. Random
```
Doctor's winrate: 50.00%
Plague's steps to win: 23.96 +/- 12.53
Doctor's steps to win: 31.20 +/- 12.09
```

#### 10 simulation, No delay; Trial 2.24  vs. 1.5
```
Doctor's winrate: 50.00%
Plague's steps to win: 25.86 +/- 13.80
Doctor's steps to win: 31.42 +/- 12.97
```

#### 10 simulation, No delay; Trial 1.5 vs. 2.24
```
Plague's steps to win: 24.00 +/- 11.17
Doctor's steps to win: 28.56 +/- 10.47
```

So... 2.24 is still better?
