
# Simulation

* gen16/train/plague-trial2.pth.24
  * (TRIAL, DELAY) = (12, 0) x 2 (per container) x 3 (#containers)
  * setups-sim: 400/4 new for each

## Collected

* 144502 (gen18 sim) + 316193 (gen17 sim)

# Train

## Trial 1

Interesting to compare to gen17/trial1. It is not clear why gen17/trial1 cannot retrieve common knowledge from the game data. test/policy and test/value goes up immediately.

The differences are (in this trial 1)

* Larger dataset (training 402660 vs. 270000, testing: 57540 vs. 42000)
* Newly collected game play data are simulated with higher trial count (12 vs. 10)
* New dataset processing: randomly shuffled before splitted

Training/testing with game data purely seems to saturate at some point. This makes sense because we don't think the game data are 100% correct but tsume data are.

