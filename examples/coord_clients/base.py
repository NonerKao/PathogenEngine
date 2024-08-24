from abc import ABC, abstractmethod
from constant import *

class Agent(ABC):
    def __init__(self, args):
        self.result = None
        if args.record is not None:
            # regex pattern is like 4, (389, 1, 4)\+, for now
            self.record = open(args.record, 'wb')
        else:
            self.record = None
        self.verbose = args.verbose

    def play(self):
        data = self.s.recv(CODE_DATA+S)
        if self.record is not None:
            self.record.write(data)
        if self.verbose:
            print(data[0:4])
        # There are three groups of status code:
        #    1. Errors
        #    2. Starters
        #    3. Closers
        # they all returns usable self.action
        self.analyze(data)
        if self.record is not None:
            self.record.write(self.action.to_bytes(1, 'little'))
        if data[0:4] in (b'Ix01', b'Ix03', b'Ix07', b'Ix08', b'Ix09', b'Ix0a'):
            self.s.sendall(bytes([self.action]))
            if self.verbose:
                print(self.action)
        elif data[0] != ord('I'):
            self.s.sendall(bytes([self.action]))
            if self.verbose:
                print(self.action)
        else:
            if data[0:4] in (b'Ix00', b'Ix02'):
                pass
            elif data[0:4] == b'Ix04':
                if self.verbose:
                    print("win!")
                self.result = b'Ix04'
                return False;
            elif data[0:4] == b'Ix05':
                if self.verbose:
                    print("lose!")
                return False;
            elif data[0:4] == b'Ix06':
                if self.verbose:
                    print("disconnected")
                return False;
            else:
                print("What is this?")
                panic()

        return True

    @abstractmethod
    def analyze(self, data):
        pass
