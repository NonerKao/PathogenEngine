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
        self.replay = []
        self.tail = 0
        self.head = 0

    def analyze(self, data):
        if ord('E') == data[-4]:
        # Exclude current self.action from self.map
            if self.head < self.tail:
                self.action = self.replay
            if self.action in self.map:
                self.map.remove(self.action)
        elif b'Ix02' == data[-4:]:
            return
        elif b'Ix01' == data[-4:]:
            self.replay.append(self.action)
            self.tail = self.tail + 1
            return
        else:
            self.map = self.fixmap.copy()
        self.action = random.choice(self.map)
        if self.count > 100:
            panic
        else:
            self.count = self.count + 1
            print(self.action)
            print(self.map)
            print(data[-4:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A Random Agent for Pathogen')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')
    
    args = parser.parse_args()
    d = RandomAgent(args.side)
    while d.play():
        continue
