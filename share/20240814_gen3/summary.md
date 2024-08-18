
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

```
Doctor's winrate: 50.00%
Plague's steps to win: 30.82 +/- 17.67
Doctor's steps to win: 42.38 +/- 16.15

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
Doctor's winrate: 40.00%
Plague's steps to win: 25.33 +/- 9.18
Doctor's steps to win: 37.00 +/- 12.28
play/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-random
Doctor's winrate: 35.00%
Plague's steps to win: 36.23 +/- 23.73
Doctor's steps to win: 45.14 +/- 16.77
```

# Play 3

Win rate distribution is interesting. It is observed that inversely trained
ones are indeed worse, for the first time!

| Doctor\Plague | Random | gen0 | gen2 (inversely trained) | gen3 |
| ------------- | ------ | ---- | ------------------------ | ---- | 
| Random        | 56%    | 61%  | 71%                      | 72%  |
| gen0          | 45%    | 57%  | 71%                      | 72%  |
| gen2          | 34%    | 38%  | 71%                      | 60%  |
| gen3          | 43%    | 57%  | 75%                      | 75%  |

## Raw data

```
===
play3/records
Doctor's winrate: 59.88%
Plague's steps to win: 27.22 +/- 13.46
Doctor's steps to win: 33.28 +/- 12.45
===
play3/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-#mnt#20240814_gen3#train#plague.pth.8
Doctor's winrate: 75.00%
Plague's steps to win: 24.76 +/- 15.15
Doctor's steps to win: 32.56 +/- 11.59
Stay rate: 0%
Bad policy rate: 5.72698%
Invalidity rate: 14.5648%
Stay rate: 0.00682594%
Bad policy rate: 9.16724%
Invalidity rate: 16.6416%
===
play3/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-#mnt#20240813_gen2#train#plague.pth.500
Doctor's winrate: 75.00%
Plague's steps to win: 25.64 +/- 14.91
Doctor's steps to win: 33.07 +/- 11.12
Stay rate: 0%
Bad policy rate: 3.90938%
Invalidity rate: 23.0474%
Stay rate: 0%
Bad policy rate: 8.86769%
Invalidity rate: 17.2754%
===
play3/records/d-#mnt#20240813_gen2#train#doctor.pth.500-p-#mnt#20240814_gen3#train#plague.pth.8
Doctor's winrate: 60.00%
Plague's steps to win: 30.05 +/- 12.97
Doctor's steps to win: 34.73 +/- 14.90
Stay rate: 0%
Bad policy rate: 6.0599%
Invalidity rate: 14.8658%
Stay rate: 0%
Bad policy rate: 5.10823%
Invalidity rate: 22.4509%
===
play3/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-#mnt#20240812_gen0#train#plague.pth
Doctor's winrate: 57.00%
Plague's steps to win: 28.21 +/- 12.41
Doctor's steps to win: 33.54 +/- 13.14
Stay rate: 0%
Bad policy rate: 23.1469%
Invalidity rate: 11.4462%
Stay rate: 0.00680689%
Bad policy rate: 8.41331%
Invalidity rate: 17.3916%
===
play3/records/d-#mnt#20240813_gen2#train#doctor.pth.500-p-#mnt#20240812_gen0#train#plague.pth
Doctor's winrate: 38.00%
Plague's steps to win: 32.81 +/- 12.21
Doctor's steps to win: 34.26 +/- 13.00
Stay rate: 0%
Bad policy rate: 23.9439%
Invalidity rate: 11.5892%
Stay rate: 0.00678426%
Bad policy rate: 4.77612%
Invalidity rate: 22.076%
===
play3/records/d-random-p-#mnt#20240814_gen3#train#plague.pth.8
Doctor's winrate: 72.00%
Plague's steps to win: 32.43 +/- 18.85
Doctor's steps to win: 31.11 +/- 10.11
Stay rate: 0%
Bad policy rate: 5.54925%
Invalidity rate: 14.1895%
===
play3/records/d-#mnt#20240812_gen0#train#doctor.pth-p-random
Doctor's winrate: 45.00%
Plague's steps to win: 24.38 +/- 13.32
Doctor's steps to win: 34.58 +/- 10.89
Stay rate: 0.0147951%
Bad policy rate: 19.8698%
Invalidity rate: 11.5476%
===
play3/records/d-random-p-random
Doctor's winrate: 56.00%
Plague's steps to win: 24.73 +/- 18.48
Doctor's steps to win: 33.50 +/- 11.66
===
play3/records/d-#mnt#20240812_gen0#train#doctor.pth-p-#mnt#20240813_gen2#train#plague.pth.500
Doctor's winrate: 71.00%
Plague's steps to win: 28.86 +/- 12.63
Doctor's steps to win: 34.56 +/- 12.32
Stay rate: 0%
Bad policy rate: 4.36862%
Invalidity rate: 23.5776%
Stay rate: 0.0129274%
Bad policy rate: 20.4641%
Invalidity rate: 11.4795%
===
play3/records/d-random-p-#mnt#20240813_gen2#train#plague.pth.500
Doctor's winrate: 71.00%
Plague's steps to win: 27.69 +/- 13.69
Doctor's steps to win: 33.75 +/- 13.99
Stay rate: 0%
Bad policy rate: 4.00765%
Invalidity rate: 22.9983%
===
play3/records/d-#mnt#20240813_gen2#train#doctor.pth.500-p-random
Doctor's winrate: 34.00%
Plague's steps to win: 23.42 +/- 11.42
Doctor's steps to win: 32.47 +/- 14.03
Stay rate: 0.0166694%
Bad policy rate: 4.52575%
Invalidity rate: 21.1535%
===
play3/records/d-#mnt#20240812_gen0#train#doctor.pth-p-#mnt#20240814_gen3#train#plague.pth.8
Doctor's winrate: 72.00%
Plague's steps to win: 26.50 +/- 12.73
Doctor's steps to win: 35.03 +/- 11.87
Stay rate: 0%
Bad policy rate: 5.74292%
Invalidity rate: 14.3334%
Stay rate: 0.0129685%
Bad policy rate: 20.4772%
Invalidity rate: 11.6846%
===
play3/records/d-random-p-#mnt#20240812_gen0#train#plague.pth
Doctor's winrate: 61.00%
Plague's steps to win: 29.26 +/- 13.16
Doctor's steps to win: 29.18 +/- 11.54
Stay rate: 0%
Bad policy rate: 22.5078%
Invalidity rate: 11.5126%
===
play3/records/d-#mnt#20240812_gen0#train#doctor.pth-p-#mnt#20240812_gen0#train#plague.pth
Doctor's winrate: 57.00%
Plague's steps to win: 25.84 +/- 10.42
Doctor's steps to win: 30.67 +/- 9.42
Stay rate: 0%
Bad policy rate: 22.9882%
Invalidity rate: 11.3335%
Stay rate: 0%
Bad policy rate: 19.9115%
Invalidity rate: 11.5339%
===
play3/records/d-#mnt#20240813_gen2#train#doctor.pth.500-p-#mnt#20240813_gen2#train#plague.pth.500
Doctor's winrate: 71.00%
Plague's steps to win: 26.10 +/- 9.69
Doctor's steps to win: 36.54 +/- 15.36
Stay rate: 0%
Bad policy rate: 4.43287%
Invalidity rate: 23.2224%
Stay rate: 0.00670646%
Bad policy rate: 4.99631%
Invalidity rate: 23.2043%
===
play3/records/d-#mnt#20240814_gen3#train#doctor.pth.8-p-random
Doctor's winrate: 43.00%
Plague's steps to win: 26.19 +/- 11.70
Doctor's steps to win: 32.37 +/- 12.77
Stay rate: 0%
Bad policy rate: 8.08267%
Invalidity rate: 16.4154%
```
