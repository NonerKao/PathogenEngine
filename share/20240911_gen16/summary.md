
# Simulation

Skipped. All apply tsume_generator.

# Train

## Train 1: new

* Training: 300000 (extra tsume)
* Testing: 24000 (extra tsume)

## Train 2: new

* Training: 480000 (extra tsume)
* Testing: 57000 (extra tsume)

Looks very good!

### Play

#### Baseline

```
Doctor's winrate: 53.00%
Plague's steps to win: 27.38 +/- 13.77
Doctor's steps to win: 32.45 +/- 11.09
```

#### No reading, no delay
```
Doctor's winrate: 48.00%
Plague's steps to win: 23.00 +/- 11.38
Doctor's steps to win: 29.62 +/- 11.48
```

#### reading 50, no delay, but this is before commit 7d5f4fd
```
Doctor's winrate: 53.00%
Plague's steps to win: 21.30 +/- 10.48
Doctor's steps to win: 32.57 +/- 13.30
```
#### reading 10, no delay, but this is after commit 7d5f4fd
```
Doctor's winrate: 42.00%
Plague's steps to win: 22.93 +/- 10.92
Doctor's steps to win: 32.57 +/- 11.59
```
