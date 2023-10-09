import socket
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, fraction):
        self.fraction = fraction
        if fraction == "Doctor":
            self.port = 6241
            self.action = 40
        elif fraction == "Plague":
            self.port = 3698
            self.action = 80
        else:
            raise ValueError("Unknown fraction!")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', self.port))

    def play(self):
        data = self.s.recv(154)
        action = self.get_coord(data)
        self.s.sendall(bytes([action]))

    @abstractmethod
    def get_coord(self, data):
        pass
