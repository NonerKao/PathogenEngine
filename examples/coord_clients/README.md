
= Quick Examples = 

Verbosely show the game transactions, record detailed status in test.bin, use the random agent, as Doctor
```
$ python main.py -v -r test.bin --seed $(date +%N) -t Random -s Doctor >doc.log
```

Use the reinforcement agent, as Plague, and enable online training, running 10 games in a roll
```
$ python main.py -t Reinforcement -s Plague --online-training -b 10 & \
  python main.py -r doc_temp.bin -t Random -s Doctor -b 10
```

= Reinforcement Learning Loop =

We imitate the AlphaZero approach to construct the training loop, which has three stages:

* Simulation: Use the model as a way to browse the game tree only. Along the way, apply MCTS methods to **label** the policies and calculate the value of certain game state. Also, since I want to check if the network really understand the rule, I also produce the third head in the end of the model: the understanding value to see if a move is valid or not.
* Train: Use the dataset applied previously.
* Evaluation: Setup the arena and let them fight!

Repeat the above process a few times, and we should see some models being stronger and stronger...
