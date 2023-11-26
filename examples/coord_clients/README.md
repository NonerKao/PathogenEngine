
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


