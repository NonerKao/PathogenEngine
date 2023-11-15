
## Generate Dataset

```
$ ./gen_dataset.sh DIR
```

## The Usage of the Driver

Currently, this model takes **Q**, the full game state (unfortunately now it lacks the knowledge of whose turn it is right now...), and a **Position** as the input. The output is to determine if the compbination is a valid one. That is, the return status code indicates that it is a legal position and thus forms a sub-step of a composite move.

### Train

```
$ python ./driver.py -t -v -d xxx.bin
```

### Validate

```
$ python ./driver.py -v -d xxx.bin -m yyy.pth
```
