import random
import sys
from base import Agent
from utils import output
from constant import *

class QAgent(Agent):
    def __init__(self, f, s):
        super().__init__(f)
        self.fixmap = list(range(36)) + list(range(100, 125))
        self.map = self.fixmap
        self.action = -1
        self.num_candidate = 0
        self.candidate = None
        self.s = s
        while self.play():
            continue
        if self.record is not None:
            self.record.close()

    def analyze(self, data):
        # Maintain the map
        if ord('E') == data[0]:
            print("Well, fall back")
            self.map.remove(self.action)
            self.candidate = None
            # print("We only implement setmap now")
            # output(data)
            # sys.exit(255)
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05'):
            # Shouldn't be used after an action commited and before the opponant's next move
            self.action = 255
            return
        elif data[0:4] in (b'Ix01', b'Ix03', b'Ix0b'):
            # ~~self.map = self.fixmap.copy()~~
            # get candidate move from query
            self.action = 255
            self.s.sendall(bytes([self.action]))
            self.num_candidate = int.from_bytes(self.s.recv(1), byteorder='big')
            #print(self.num_candidate)
            self.candidate = self.s.recv(self.num_candidate)
            #print(self.candidate)
            code = self.s.recv(CODE_DATA)
            while code not in (b'Wx00'):
                print("Unexpected query results!")
                sys.exit(255)
        # Make next action
        if self.candidate != None:
            self.action = random.choice(self.candidate)
        elif len(self.map) != 0:
            self.action = random.choice(self.map)
        else:
            print("This doesn't make any sense. Check it!")
            output(data)
            sys.exit(255)
