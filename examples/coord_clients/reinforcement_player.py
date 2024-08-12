import random
import sys
from base import Agent
from utils import output
from constant import *
import torch
import numpy as np

class RLPlayer(Agent):
    def __init__(self, args):
        super().__init__(args)
        torch.set_default_dtype(torch.float32)
        self.model = torch.load(args.model)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        if not torch.cuda.is_available():
            print('Warning: Use CPU')

        self.action = -1
        self.candidate = None

        # statistics
        self.num_stay = 0
        self.num_miss_policy = 0
        self.num_miss_valid = 0
        self.num_total = 0

        while self.play():
            continue
        if self.record is not None:
            self.record.close()

    def send_special(self, action):
        self.s.sendall(bytes([action]))
        self.num_candidate = int.from_bytes(self.s.recv(1), byteorder='big')
        self.candidate = self.s.recv(self.num_candidate)
        code = self.s.recv(CODE_DATA)
        while code not in (b'Wx00'):
            print("Unexpected query results:", self.num_candidate, "; candidates:", self.candidate);
            sys.exit(255)

    def analyze(self, data):
        self.num_total += 1
        if ord('E') == data[0]:
            print("Something went wrong!")
            output(data)
            sys.exit(255)
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05'):
            # Shouldn't be used after an action commited and before the opponant's next move
            self.action = 255

            if data[0:4] in (b'Ix00'):
                self.num_stay += 1
            elif data[0:4] in (b'Ix04', b'Ix05'):
                print(f'{self.num_stay}, {self.num_miss_policy}, {self.num_miss_valid}, {self.num_total}')
            return
        elif data[0:4] in (b'Ix01', b'Ix03'):
            self.send_special(QUERY)

        self.action = None
        # Make next action
        # We hope the model can learn what to go next. Pick top 3 here.
        self.state = np.frombuffer(data[4:], dtype=np.uint8)
        self.state = torch.from_numpy(np.copy(self.state)).float().unsqueeze(0).to(self.device)
        policy, valid, value = self.model(self.state)
        p_value, p_index = torch.topk(policy, 3)

        # Calculate policy miss statistics
        real_candidate = []
        for i in p_index.squeeze(0):
            item = i.item()
            if item < BOARD_POS and item in list(self.candidate):
                if not self.action:
                    self.action = item
                real_candidate.append(i)
            if item >= BOARD_POS and item + MAP_POS_OFFSET in list(self.candidate):
                if not self.action:
                    self.action = item
                real_candidate.append(i + MAP_POS_OFFSET)

        # Calculate valid miss statistics
        for i, v in enumerate(valid.squeeze(0)):
            if v > 0.5:
                item = i
                if item < BOARD_POS and item in list(self.candidate):
                    continue
                if item >= BOARD_POS and item + MAP_POS_OFFSET in list(self.candidate):
                    continue
                self.num_miss_valid += 1
                break

        if real_candidate:
            self.action = random.choice(real_candidate)
        elif self.candidate != None:
            self.num_miss_policy += 1
            self.action = random.choice(self.candidate)
        else:
            print("This doesn't make any sense. Check it!")
            output(data)
            sys.exit(255)
