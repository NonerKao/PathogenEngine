import random
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
        if self.record is not None:
            self.record.write(self.action.to_bytes(1, 'little'))
