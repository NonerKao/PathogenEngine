
# Simulation

* gen13/train/plague-trial3.pth.4
  * (TRIAL, DELAY) = (50, 15)
  * 606 games, setups-sim1
* gen13/train/plague-trial3.pth.4
  * (TRIAL, DELAY) = (20, 13), (20,12)
  * 600 setups-sim2
* gen13/train/plague-trial3.pth.4
  * (TRIAL, DELAY) = (40, 14)
  * 600 setups-sim3
* gen13/train/plague-trial3.pth.4
  * (TRIAL, DELAY) = (20, 9), (20, 10), (20, 11), (20, 12)
  * 500 setups-sim4

Collected 78365 game entries, and 22004 tsume entries.

# Train

## Trial 1: from gen13/train/plague-trial3.pth.4

* Training: 18000 (tsume) + 72000 (game)
* Testing: 3000 (tsume) + 6000 (game)
* weight of heads: (0.25, 0.25, 0.25, 0.25)


## Trial 2: from gen13/train/plague-trial3.pth.4

* Training: 60000 (new tsume) + 60000 (game)
* Testing: 12000 (tsume from the games) + 12000 (game)
* weight of heads: (0.25, 0.25, 0.25, 0.25)

## Trial 3: started new one

* Training: 500000 (new tsume, from B+7 to W+16)
* Testing: 48000 (new tsume, from B+7 to W+16) + 48000 (game)
* weight of heads: (0.25, 0.25, 0.25, 0.25)

* play 1 on plague.pth.7: 50.33 % ... not impressive
* play 2 on plague.pth.57 (there was a local valid minimum): 57.00% vs 53.00% ... not impressive.

