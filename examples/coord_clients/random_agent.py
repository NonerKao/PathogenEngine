import random
import argparse
from base import Agent

class RandomAgent(Agent):
    def __init__(self, f):
        super().__init__(f)
        self.fixmap = [i for i in range(57)]
        self.map = self.fixmap
        self.action = -1
        self.count = 0

    def analyze(self, data):
        print(data[-4:])
        if ord('E') == data[-4]:
	    # Exclude current self.action from self.map
            self.map.remove(self.action)
        elif b'Ix02' == data[-4:] or b'Ix04' == data[-4:] or b'Ix05' == data[-4:]:
            # Shouldn't be used after an action commited and before the opponant's next move
            self.action = -1
            return
        elif data[-4:] == b'Ix03' or data[-4:] == b'Ix01':
            # every move needs a new map
            self.map = self.fixmap.copy()
        self.action = random.choice(self.map)
        if self.count > 10000:
            panic
        else:
            self.count = self.count + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A Random Agent for Pathogen')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')
    
    args = parser.parse_args()
    d = RandomAgent(args.side)
    while d.play():
        continue
