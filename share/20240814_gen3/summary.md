
# Summary

Accepting the fact that the current result is worse than random,
The next round will be based on `20240812_gen0`

# Simulation

* DELAY = 0
* TRIAL = 100
* Others
  * Collect 100K entries for each fractions
  * Mix d-20240812_gen0-p-empty and d-empty-p-20240812_gen0

# Train

* LR = 0.0001
* Network
  * remains unchanged
* Weights of Losses
  * Remove DELTA
  * Reduce the final layers of validity to 1
* Others
  * Smaller Batch: 50
  * Inner Epoch: 1
  * Outer Epoch: 2
  * kfold: 4

# Play

## Setups

A new set of 5 sgfs, and each are played 4 times.

## Result

> in Doctor's win rate.

> Random in 6533 seed.

| Doctor\Plague | Random | gen0 | gen3 2/8 | gen3 4/8 | gen3 6/8 | gen3 8/8 |
| ------------- | ------ | ---- | -------- | -------- | -------- | -------- | 
| Random        |        |      | 70%      | 85%      | 60%      | 75%      |
| gen0          | N/A    | N/A  | N/A      | 50%      |          | 55%      |
| gen3 2/8      | 15%    | N/A  | N/A      | N/A      | N/A      | N/A      | 
| gen3 4/8      | 25%    | 40%  | N/A      | N/A      | N/A      | N/A      | 
| gen3 6/8      | 35%    | N/A  | N/A      | N/A      | N/A      | N/A      | 
| gen3 8/8      | 35%    | 55%  | N/A      | N/A      | N/A      | N/A      | 

## average ... ???

Doctor's winrate: 50.00%
Plague's steps to win: 30.82 +/- 17.67
Doctor's steps to win: 42.38 +/- 16.15

```
play/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-#mnt#20240812_gen0#train#plague.pth
Doctor's winrate: 55.00%
Plague's steps to win: 31.44 +/- 11.13
Doctor's steps to win: 41.82 +/- 6.90
play/records/d-random-p-#mnt#20240814_gen3#train#plague.pth.4
Doctor's winrate: 85.00%
Plague's steps to win: 39.00 +/- 21.07
Doctor's steps to win: 34.00 +/- 13.82
play/records/d-random-p-#mnt#20240814_gen3#train#plague.pth.6
Doctor's winrate: 60.00%
Plague's steps to win: 25.75 +/- 15.71
Doctor's steps to win: 32.50 +/- 18.37
play/records/d-random-p-#mnt#20240814_gen3#train#plague.pth.8
Doctor's winrate: 75.00%
Plague's steps to win: 24.20 +/- 10.55
Doctor's steps to win: 40.40 +/- 15.05
play/records/d-#mnt#20240814_gen3#train#doctor.pth.4-p-random
Doctor's winrate: 25.00%
Plague's steps to win: 29.13 +/- 14.61
Doctor's steps to win: 33.20 +/- 10.06
play/records/d-#mnt#20240812_gen0#train#doctor.pth-p-#mnt#20240814_gen3#train#plague.pth.4
Doctor's winrate: 50.00%
Plague's steps to win: 38.20 +/- 26.23
Doctor's steps to win: 52.60 +/- 16.68
play/records/d-random-p-#mnt#20240814_gen3#train#plague.pth.2
Doctor's winrate: 70.00%
Plague's steps to win: 36.67 +/- 31.56
Doctor's steps to win: 47.71 +/- 15.92
play/records/d-#mnt#20240814_gen3#train#doctor.pth.2-p-random
Doctor's winrate: 15.00%
Plague's steps to win: 33.59 +/- 17.96
Doctor's steps to win: 30.00 +/- 8.72
play/records/d-#mnt#20240814_gen3#train#doctor.pth.6-p-random
Doctor's winrate: 35.00%
Plague's steps to win: 27.46 +/- 12.89
Doctor's steps to win: 47.14 +/- 18.43
play/records/d-#mnt#20240812_gen0#train#doctor.pth-p-#mnt#20240814_gen3#train#plague.pth.8
Doctor's winrate: 55.00%
Plague's steps to win: 25.44 +/- 13.33
Doctor's steps to win: 60.00 +/- 9.38
play/records/d-#mnt#20240814_gen3#train#doctor.pth.4-p-#mnt#20240812_gen0#train#plague.pth
```
Doctor's winrate: 40.00%
Plague's steps to win: 25.33 +/- 9.18
Doctor's steps to win: 37.00 +/- 12.28
play/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-random
Doctor's winrate: 35.00%
Plague's steps to win: 36.23 +/- 23.73
Doctor's steps to win: 45.14 +/- 16.77

# Play 2

...
