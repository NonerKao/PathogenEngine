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

    def play(self):
        data = self.s.recv(391)
        self.analyze(data)
        if data[-4:] == b'Ix01' or data[-4:] == b'Ix03':
            self.s.sendall(bytes([self.action]))
        elif data[-4:] == b'Ix02':
            pass
        elif data[-4:] == b'Ix04':
            print("win!")
            return False;
        elif data[-4:] == b'Ix05':
            print("lose!")
            return False;
        else:
            #print("Your foul!")
            self.s.sendall(bytes([self.action]))

        return True

    @abstractmethod
    def analyze(self, data):
        pass
