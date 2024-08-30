
# Simulation

Merge the plague data from gen3 to gen5.

# Train

* TRAINING_BATCH_UNIT = 200
* TRAINING_INNER_EPOCH = 4
* TRAINING_OUTER_EPOCH = 4
* LEARNING_RATE = 0.0001
* KFOLD = 8
* ALPHA = 0.30
* BETA = 0.30
* GAMMA = 0.40
* Network structure change
  * RES_SIZE = 12 -> 18
  * RES_INPUT_SIZE = 84 -> 90

Use a small test set (6800 cases). Observe a local minimum around epoch 14,
where the Kfold wasn't fully explored yet.

# Play

## Setups

New a set of 5 sgfs, play 20 times each.
> in Doctor's win rate.
> Random in 6533 seed.

Random-random: 60%.

## Result records 1st, 2nd

> Shit, I get the order reverted...

Looks terrible.

## Result records 3rd

Compare epoch 15 and 128 and see what will happen.

