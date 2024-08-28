
# Simulation

Enable simulated play, which should be stronger, but not observing such an improvement.

## Setups

New a set of 100 sgfs, play 1 times each.
> in Doctor's win rate.
> Random in 6533 seed.

Collect a few set of dataset entries with different trial/delay settings.

# Train

## Trial 0

Tweak the network into (12,144) and see what will happen.
Also, the policy is weighted with validity (understanding).

TRAINING_BATCH_UNIT = 10
TRAINING_INNER_EPOCH = 1
TRAINING_OUTER_EPOCH = 2

LEARNING_RATE = 0.0001
KFOLD = 6

ALPHA = 0.20
BETA = 0.30
GAMMA = 0.50

## Trial 1: No Diff

## Trial 2: Diff to trail 0

KFOLD = 3

## Trial 3: Diff to trail 0
TRAINING_BATCH_UNIT = 20
LEARNING_RATE = 0.0002

Worse, why?

## Trial 4: Diff to trail 0
TRAINING_BATCH_UNIT = 20

> This was overwrittern.

## Trial 5: Diff to trail 0
TRAINING_BATCH_UNIT = 60

Also increase the dataset size. from 66000 to 288000, and testing from 4020 to 10800.

This looks far better. Testing set reports lower loss, but train/val seems not too good but that's OK.

## Trial 6: Diff to trail 0
TRAINING_BATCH_UNIT = 120

Worse than 60. Also, considering we are using `sum` for the loss to validity head, consider changing this...
