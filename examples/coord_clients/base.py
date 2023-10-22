import socket
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, fraction):
        self.fraction = fraction
        if fraction == "Doctor":
            self.port = 6241
        elif fraction == "Plague":
            self.port = 3698
        else:
            raise ValueError("Unknown fraction!")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', self.port))
        self.s.setblocking(1)
        self.s.settimeout(None)

    def play(self):
        data = self.s.recv(391)
        print(data[-4:])
        # if b'' == data:
        #    return True
        if data[-4:] in (b'Ix01', b'Ix03'):
            self.analyze(data)
            self.s.sendall(bytes([self.action]))
            print(self.action)
        elif data[-4:] in (b'Ix00', b'Ix02'):
            pass
        elif data[-4:] == b'Ix04':
            print("win!")
            return False;
        elif data[-4:] == b'Ix05':
            print("lose!")
            return False;
        elif data[-4:] == b'Ix06':
            print("disconnected")
            return False;
        else:
            #print("Your foul!")
            self.analyze(data)
            self.s.sendall(bytes([self.action]))
            print(self.action)

        return True

    @abstractmethod
    def analyze(self, data):
        pass
