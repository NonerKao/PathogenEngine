import socket
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, a):
        self.fraction = a.side
        if self.fraction == "Doctor":
            self.port = 6241
        elif self.fraction == "Plague":
            self.port = 3698
        else:
            raise ValueError("Unknown fraction!")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', self.port))
        self.s.setblocking(1)
        self.s.settimeout(None)
        if a.record is not None:
            # regex pattern is like ((391, 1)\+ 391)*
            self.record = open(a.record, 'wb')
        else:
            self.record = None
        self.verbose = a.verbose

    def play(self):
        data = self.s.recv(391)
        if self.record is not None:
            self.record.write(data)
        if self.verbose:
            print(data[-4:])
        # if b'' == data:
        #    return True
        if data[-4:] in (b'Ix01', b'Ix03'):
            self.analyze(data)
            self.s.sendall(bytes([self.action]))
            if self.verbose:
                print(self.action)
        elif data[-4:] in (b'Ix00', b'Ix02'):
            pass
        elif data[-4:] == b'Ix04':
            if self.verbose:
                print("win!")
            return False;
        elif data[-4:] == b'Ix05':
            if self.verbose:
                print("lose!")
            return False;
        elif data[-4:] == b'Ix06':
            if self.verbose:
                print("disconnected")
            return False;
        else:
            self.analyze(data)
            self.s.sendall(bytes([self.action]))
            if self.verbose:
                print(self.action)

        return True

    @abstractmethod
    def analyze(self, data):
        pass
