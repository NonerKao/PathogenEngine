import random
import argparse
from base import Agent

class RandomAgent(Agent):
    def __init__(self, f):
        super().__init__(f)
        self.fixmap = list(range(36)) + list(range(100, 125))
        self.map = self.fixmap
        self.action = -1
        self.count = 0

    def analyze(self, data):
        if ord('E') == data[-4]:
	    # Exclude current self.action from self.map
            self.map.remove(self.action)
        elif data[-4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05'):
            # Shouldn't be used after an action commited and before the opponant's next move
            self.action = -1
            return
        elif data[-4:] in (b'Ix01', b'Ix03'):
            # every move needs a new map
            self.map = self.fixmap.copy()
        if len(self.map) != 0:
            self.action = random.choice(self.map)
        else:
            print("This doesn't make any sense. Check it. Resign...")
            output(data)
            data[-4:] = b'Ix05'

def output(data):
    for i in range(6):
        for j in range(6):
            print(f"({i}, {j})", end=' ')
            print(data[i*6*9 + j*9 : i*6*9 + j*9 + 9])

    print()

    # The second loop outputs the 5*5*2 bytes
    for i in range(5):
        for j in range(5):
            print(f"({i-2}, {j-2})", end=' ')
            print(data[i*5*2 + j*2 : i*5*2 + j*2 + 2])

    print()
    print(data[374:387])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A Random Agent for Pathogen')
    parser.add_argument('--seed', type=str, help='Seed for the random number generator')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        print("Seed is set to:", args.seed)

    d = RandomAgent(args.side)
    while d.play():
        continue
