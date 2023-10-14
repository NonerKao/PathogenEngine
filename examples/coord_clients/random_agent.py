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

        self.is_replaying = False
        self.replay = []
        self.tail = 0
        self.head = 0

    def analyze(self, data):
        if b'Ex0B' == data[-4:] or b'Ex0C' == data[-4:] or b'Ex0D' == data[-4:]:
	    # SetMarker finished but not passed the check
            print(data[-17:-4])
        elif ord('E') == data[-4]:
	    # Exclude current self.action from self.map
            if self.head < self.tail:
                self.action = self.replay[self.head]
                self.head = self.head + 1
                return
            self.map.remove(self.action)
        elif b'Ix02' == data[-4:]:
            self.replay = []
            self.tail = 0
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
