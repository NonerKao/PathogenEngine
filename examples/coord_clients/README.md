
# Quick Examples

Verbosely show the game transactions, record detailed status in test.bin, use the random agent, as Doctor
```
$ python main.py -v -r test.bin --seed $(date +%N) -t Random -s Doctor >doc.log
```

Use the reinforcement agent, as Plague to generate data for training
```
$ python main.py -t Reinforcement -s Plague -b 10 -m test.pth & \
  python main.py -r doc_temp.bin -t Random -s Doctor -b 10 --seed 5566
```

# Reinforcement Learning Loop

We imitate the AlphaZero approach to construct the training loop, which has three stages:

* Simulation: Use the model as a way to browse the game tree only. Along the way, apply MCTS methods to **label** the policies and calculate the value of certain game state. Also, since I want to check if the network really understand the rule, I also produce the third head in the end of the model: **valid** vector to see the validity of every move.
* Train: Use the dataset applied previously.
* Evaluation: Setup the arena and let them fight!

Repeat the above process a few times, and we should see some models being stronger and stronger...

## Hacking Simulation

You may tweak the `DELAY_UNIT` and `TRIAL_UNIT`. The former is set to delay the simulation for a few steps, because of the heuristic that the decision made near the end game will be more informational. The latter is to define how heavy the simulation will be.

We implement the MCTS algorithm based on Ch.6 of [the book](https://www.books.com.tw/products/0010881844). Roughly, there is three events to update the visited number and value of a node. Mostly it is a recursive call, and the last two are the terminations. They are:

* When the child nodes of the current node is known. Go to the next node whose PUCT value is the maximum.
* When we hasn't known what will be the child nodes of the current node. We need to expand the candidates of available actions. Update the w (accumilated weight) and n (#trials) and Return the value (using the model) of this node.
* When we reach to an end-game node. Update the w (accumilated weight) and n (#trials) and Return the value (using the model) of this node based on winning or losing.

## Hacking Training

## Hacking Playing
