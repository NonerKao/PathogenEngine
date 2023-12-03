import socket
from abc import ABC, abstractmethod
from constant import *
import random

def random_transform(data):
    trans = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
    transformed_data = bytearray(data)

    rotation = trans//2
    if rotation == 0:
        pass
    elif rotation == 1:
        # counter clock-wise 90 degrees
        for i in range(0, BOARD_SIZE//2):
            for j in range(0, BOARD_SIZE//2):
                src_offset = CODE_DATA + i*BOARD_SIZE*BOARD_ATTR + j*BOARD_ATTR
                dst1_offset = CODE_DATA + (BOARD_SIZE-1-j)*BOARD_SIZE*BOARD_ATTR + i*BOARD_ATTR
                dst2_offset = CODE_DATA + (BOARD_SIZE-1-i)*BOARD_SIZE*BOARD_ATTR + (BOARD_SIZE-1-j)*BOARD_ATTR
                dst3_offset = CODE_DATA + j*BOARD_SIZE*BOARD_ATTR + (BOARD_SIZE-1-i)*BOARD_ATTR
                tmp = transformed_data[src_offset:src_offset+BOARD_ATTR]
                transformed_data[src_offset:src_offset+BOARD_ATTR] = transformed_data[dst1_offset:dst1_offset+BOARD_ATTR]
                transformed_data[dst1_offset:dst1_offset+BOARD_ATTR] = transformed_data[dst2_offset:dst2_offset+BOARD_ATTR]
                transformed_data[dst2_offset:dst2_offset+BOARD_ATTR] = transformed_data[dst3_offset:dst3_offset+BOARD_ATTR]
                transformed_data[dst3_offset:dst3_offset+BOARD_ATTR] = tmp
        for i in range(0, MAP_SIZE//2+1):
            for j in range(0, MAP_SIZE//2):
                src_offset = CODE_DATA + BOARD_DATA + i*MAP_SIZE*MAP_ATTR + j*MAP_ATTR
                dst1_offset = CODE_DATA + BOARD_DATA + (MAP_SIZE-1-j)*MAP_SIZE*MAP_ATTR + i*MAP_ATTR
                dst2_offset = CODE_DATA + BOARD_DATA + (MAP_SIZE-1-i)*MAP_SIZE*MAP_ATTR + (MAP_SIZE-1-j)*MAP_ATTR
                dst3_offset = CODE_DATA + BOARD_DATA + j*MAP_SIZE*MAP_ATTR + (MAP_SIZE-1-i)*MAP_ATTR
                tmp = transformed_data[src_offset:src_offset+MAP_ATTR]
                transformed_data[src_offset:src_offset+MAP_ATTR] = transformed_data[dst1_offset:dst1_offset+MAP_ATTR]
                transformed_data[dst1_offset:dst1_offset+MAP_ATTR] = transformed_data[dst2_offset:dst2_offset+MAP_ATTR]
                transformed_data[dst2_offset:dst2_offset+MAP_ATTR] = transformed_data[dst3_offset:dst3_offset+MAP_ATTR]
                transformed_data[dst3_offset:dst3_offset+MAP_ATTR] = tmp
    elif rotation == 2:
        # counter clock-wise 180 degrees
        for c in range(0, BOARD_SIZE*BOARD_SIZE//2):
            i = c//BOARD_SIZE
            j = c%BOARD_SIZE
            src_offset = CODE_DATA + i*BOARD_SIZE*BOARD_ATTR + j*BOARD_ATTR
            dst_offset = CODE_DATA + (BOARD_SIZE-1-i)*BOARD_SIZE*BOARD_ATTR + (BOARD_SIZE-1-j)*BOARD_ATTR
            tmp = transformed_data[src_offset:src_offset+BOARD_ATTR]
            transformed_data[src_offset:src_offset+BOARD_ATTR] = transformed_data[dst_offset:dst_offset+BOARD_ATTR]
            transformed_data[dst_offset:dst_offset+BOARD_ATTR] = tmp
        for c in range(0, MAP_SIZE*MAP_SIZE//2):
            i = c//MAP_SIZE
            j = c%MAP_SIZE
            src_offset = CODE_DATA + BOARD_DATA + i*MAP_SIZE*MAP_ATTR + j*MAP_ATTR
            dst_offset = CODE_DATA + BOARD_DATA + (MAP_SIZE-1-i)*MAP_SIZE*MAP_ATTR + (MAP_SIZE-1-j)*MAP_ATTR
            tmp = transformed_data[src_offset:src_offset+MAP_ATTR]
            transformed_data[src_offset:src_offset+MAP_ATTR] = transformed_data[dst_offset:dst_offset+MAP_ATTR]
            transformed_data[dst_offset:dst_offset+MAP_ATTR] = tmp
    elif rotation == 3:
        # counter clock-wise 270 degrees
        for i in range(0, BOARD_SIZE//2):
            for j in range(0, BOARD_SIZE//2):
                src_offset = CODE_DATA + i*BOARD_SIZE*BOARD_ATTR + j*BOARD_ATTR
                dst1_offset = CODE_DATA + j*BOARD_SIZE*BOARD_ATTR + (BOARD_SIZE-1-i)*BOARD_ATTR
                dst2_offset = CODE_DATA + (BOARD_SIZE-1-i)*BOARD_SIZE*BOARD_ATTR + (BOARD_SIZE-1-j)*BOARD_ATTR
                dst3_offset = CODE_DATA + (BOARD_SIZE-1-j)*BOARD_SIZE*BOARD_ATTR + i*BOARD_ATTR
                tmp = transformed_data[src_offset:src_offset+BOARD_ATTR]
                transformed_data[src_offset:src_offset+BOARD_ATTR] = transformed_data[dst1_offset:dst1_offset+BOARD_ATTR]
                transformed_data[dst1_offset:dst1_offset+BOARD_ATTR] = transformed_data[dst2_offset:dst2_offset+BOARD_ATTR]
                transformed_data[dst2_offset:dst2_offset+BOARD_ATTR] = transformed_data[dst3_offset:dst3_offset+BOARD_ATTR]
                transformed_data[dst3_offset:dst3_offset+BOARD_ATTR] = tmp
        for i in range(0, MAP_SIZE//2+1):
            for j in range(0, MAP_SIZE//2):
                src_offset = CODE_DATA + BOARD_DATA + i*MAP_SIZE*MAP_ATTR + j*MAP_ATTR
                dst1_offset = CODE_DATA + BOARD_DATA + j*MAP_SIZE*MAP_ATTR + (MAP_SIZE-1-i)*MAP_ATTR
                dst2_offset = CODE_DATA + BOARD_DATA + (MAP_SIZE-1-i)*MAP_SIZE*MAP_ATTR + (MAP_SIZE-1-j)*MAP_ATTR
                dst3_offset = CODE_DATA + BOARD_DATA + (MAP_SIZE-1-j)*MAP_SIZE*MAP_ATTR + i*MAP_ATTR
                tmp = transformed_data[src_offset:src_offset+MAP_ATTR]
                transformed_data[src_offset:src_offset+MAP_ATTR] = transformed_data[dst1_offset:dst1_offset+MAP_ATTR]
                transformed_data[dst1_offset:dst1_offset+MAP_ATTR] = transformed_data[dst2_offset:dst2_offset+MAP_ATTR]
                transformed_data[dst2_offset:dst2_offset+MAP_ATTR] = transformed_data[dst3_offset:dst3_offset+MAP_ATTR]
                transformed_data[dst3_offset:dst3_offset+MAP_ATTR] = tmp

    mirror = trans%2
    if mirror == 0:
        pass
    elif mirror == 1:
        # flip against diagonal
        # for BOARD_DATA
        for i in range(0, BOARD_SIZE):
            for j in range(i+1, BOARD_SIZE):
                src_offset = CODE_DATA + i*BOARD_SIZE*BOARD_ATTR + j*BOARD_ATTR
                dst_offset = CODE_DATA + j*BOARD_SIZE*BOARD_ATTR + i*BOARD_ATTR
                tmp = transformed_data[src_offset:src_offset+BOARD_ATTR]
                transformed_data[src_offset:src_offset+BOARD_ATTR] = transformed_data[dst_offset:dst_offset+BOARD_ATTR]
                transformed_data[dst_offset:dst_offset+BOARD_ATTR] = tmp
                    
        # for MAP_DATA
        for i in range(0, MAP_SIZE):
            for j in range(i+1, MAP_SIZE):
                src_offset = CODE_DATA + BOARD_DATA + i*MAP_SIZE*MAP_ATTR + j*MAP_ATTR
                dst_offset = CODE_DATA + BOARD_DATA + j*MAP_SIZE*MAP_ATTR + i*MAP_ATTR
                tmp = transformed_data[src_offset:src_offset+MAP_ATTR]
                transformed_data[src_offset:src_offset+MAP_ATTR] = transformed_data[dst_offset:dst_offset+MAP_ATTR]
                transformed_data[dst_offset:dst_offset+MAP_ATTR] = tmp

    return bytes(transformed_data)

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
            # regex pattern is like 4, (389, 1, 4)\+, for now
            self.record = open(a.record, 'wb')
        else:
            self.record = None
        self.verbose = a.verbose

    def play(self):
        data = self.s.recv(CODE_DATA+BOARD_DATA+MAP_DATA+FLOW_DATA+TURN_DATA)
        if self.record is not None:
            self.record.write(data)
        if self.verbose:
            print(data[0:4])
        # There are three groups of status code:
        #    1. Errors
        #    2. Starters
        #    3. Closers
        # they all returns usable self.action
        data = random_transform(data)
        self.analyze(data)
        if self.record is not None:
            self.record.write(self.action.to_bytes(1, 'little'))
        if data[0:4] in (b'Ix01', b'Ix03'):
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
                return False;
            elif data[0:4] == b'Ix05':
                if self.verbose:
                    print("lose!")
                return False;
            elif data[0:4] == b'Ix06':
                if self.verbose:
                    print("disconnected")
                return False;

        return True

    @abstractmethod
    def analyze(self, data):
        pass
