I will focus on general model this time and then derive later.

# Simulation

* SIZE = 90000, hybrid
* N_GAME = 400

* TRIAL_UNIT = 80
* DELAY_UNIT = 0 
* TEMPERATURE = 2.0
* SPICE = 2

# Train

* TRAINING_BATCH_UNIT = 20
* TRAINING_INNER_EPOCH = 4
* TRAINING_OUTER_EPOCH = 10
* LR = 0.0001
* KFOLD = 3
 
* ALPHA = 0.2
* BETA = 0.2
* GAMMA = 0.6

# Play

## Setups

A new set of 4 sgfs, and each are played 10 times.

## Result

> in Doctor's win rate.
> Random in 6533 seed.

The losses during training among epochs can only serve as a refrence.
The win rate goes up and down as the training process goes.

Let's mark the best models as
* random-random: 59%
* plague.pth.best = general.pth.22 (55%)
* doctor.pth.best = general.pth.96 (65%)
