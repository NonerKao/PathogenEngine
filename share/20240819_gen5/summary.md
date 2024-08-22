Derived from 20240818_gen4.

# Simulation

* SIZE = 100000, hybrid
* N_GAME = 600

* TRIAL_UNIT = 80
* DELAY_UNIT = 0 
* TEMPERATURE = 2.0
* SPICE = 2

# Train

* TRAINING_BATCH_UNIT = 100
* TRAINING_INNER_EPOCH = 3
* TRAINING_OUTER_EPOCH = 3
* LEARNING_RATE = 0.0001
* KFOLD = 4
* ALPHA = 0.2
* BETA = 0.25
* GAMMA = 0.55

# Play

## Setups

A new set of 6 sgfs, and each are played 6 times.

## Result

> in Doctor's win rate.
> Random in 6533 seed.

Roughly positively-related between `play2/records` and `play2/records-1st`, but some still has high divergence.

Consider the timely process, I will focus on training plague for the next round.

Train the plague.pth.32 (this is the one that has the higher chance to win doctor at (1-54%). ), with only plague data.

