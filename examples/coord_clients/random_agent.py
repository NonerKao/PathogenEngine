import random
import sys
from base import Agent
from utils import output

class RandomAgent(Agent):
    def __init__(self, f):
        super().__init__(f)
        self.fixmap = list(range(36)) + list(range(100, 125))
        self.map = self.fixmap
        self.action = -1
        self.count = 0
        while self.play():
            continue
        if self.record is not None:
            self.record.close()

    def analyze(self, data):
        # Maintain the map
        if ord('E') == data[0]:
	    # Exclude current self.action from self.map
            self.map.remove(self.action)
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05'):
            # Shouldn't be used after an action commited and before the opponant's next move
            self.action = 255
            return
        elif data[0:4] in (b'Ix01', b'Ix03'):
            # every move needs a new map
            self.map = self.fixmap.copy()
        # Make next action
        if len(self.map) != 0:
            self.action = random.choice(self.map)
        else:
            print("This doesn't make any sense. Check it!")
            output(data)
            sys.exit(255)
