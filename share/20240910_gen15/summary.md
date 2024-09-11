
# Simulation

* gen13/train/plague-trial3.pth.4
  * (TRIAL, DELAY) = (40, 7)
  * 250x4 setups-sim1
* gen14/train/plague-trial4.pth.13
  * (TRIAL, DELAY) = (40, 7)
  * 250x4 setups-sim2
* gen14/train/plague-trial6.pth.3
  * (TRIAL, DELAY) = (40, 7)
  * 250x4 setups-sim3

All were interrupted.

## Collected

* game entries: 43410
* tsume entries: 11231
* extra tsume entries: 169800

# Train

## Train 1: new

* Training: 150000 (extra tsume) + 30000 (game)
* Testing: 12000 (extra tsume) + 9000 (tsume) + 12000 (game)

Doesn't look good. I would like to change the weighting.

## Train 2: new

Diff from Trial 1:

* weight of heads: (0.15, 0.25, 0.50, 0.10)

## Train 3: new

Diff from Trial 1:

* weight of heads: (0.50, 0.0, 0.50, 0.0)

Cross validation cannot learn well on policy part, for all above three.
Maybe we need a better network architecture.

Shit, the way we generate policy data, is wrong!

## Train 4: new

Diff from Trial 1:

* KFOLD = 4

Better than trial 1, take plague-trial4.pth.3

## Train 5: new

Diff from Trial 1:

* Network: 8 x 140
* KFOLD = 4


# Train Boosttrap

Shit, all of them doesn't matter. The normal game records are not to be trusted!!!
Collecting gen13/gen14/gen15 tsume records, and train on them.

Compared to trial 1

* Training: 510000
* Testing: 66000

* 9 x 160
* KFOLD 4

## Trial b1: bootstrap this

This trial performs so bad...

## Trial b2: new

* 7 x 160

## Trial b3: new

* 12 x 160

## Trial b4: new

* 6 x 160

## Trial b5: new

* 12 x 200

too long

## Trial b6: new

* 7 x 100

### play

It is weird that

#### baseline
Doctor's winrate: 53.00%
Plague's steps to win: 25.81 +/- 12.40
Doctor's steps to win: 33.62 +/- 12.18

#### zero reading
Doctor's winrate: 49.00%
Plague's steps to win: 23.08 +/- 11.36
Doctor's steps to win: 32.20 +/- 13.56

#### reading 50 simulatoins (partial, because it takes long time...)
Doctor's winrate: 60.00%
Plague's steps to win: 26.67 +/- 9.58
Doctor's steps to win: 27.33 +/- 8.06

