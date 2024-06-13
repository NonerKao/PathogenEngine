import random
import os 
import sys
from base import Agent
from utils import output
import numpy as np
import torch
from constant import *
from reinforcement_network import *

TOPK = 3

def init_model(model_name):
    torch.set_default_dtype(torch.float32)
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Warning: Starting with a new model.")
    return model

class RLAgent(Agent):
    torch.set_default_device(torch.device("cuda"))
    def __init__(self, f):
        super().__init__(f)
        self.action = 255
        self.all_transitions = torch.tensor([])

        # initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        if not torch.cuda.is_available():
            print('Warning: Use CPU')

        # We will only use this model for inference, at this phase
        self.model = init_model(f.model)
        self.model.eval()
        
        while self.play():
            continue
        if self.record is not None:
            self.record.close()
            # states, actions, rewards = self.all_transitions[:, :S].clone().to(dtype=torch.float32), self.all_transitions[:, S], self.all_transitions[:, S+1]

    def analyze(self, data):
        if ord('E') == data[0]:
            print("This doesn't make any sense. Check it!")
            sys.exit(255)
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05', b'Ix06'):
            # This won't be really sent back to the server because the code
            # indicates that we have nothing to do now. This client still set a
            # action for the sake of dataset structure.
            # Ix00 and Ix02: our turn ends with the previous (sub-)move
            # Ix04: we won!
            # Ix05: we lost...
            # Ix06: somehow, either we or the component lost the connection
            if data[0:4] not in (b'Ix00'):
                pass
            self.action = 255
            return
        elif data[0:4] in (b'Ix01', b'Ix03'):
            # get candidate move from query
            self.action = 255
            self.s.sendall(bytes([self.action]))
            self.num_candidate = int.from_bytes(self.s.recv(1), byteorder='big')
            self.candidate = self.s.recv(self.num_candidate)
            code = self.s.recv(CODE_DATA)
            while code not in (b'Wx00'):
                print("Unexpected query results!")
                sys.exit(255)

        # Make next action
        self.state = np.frombuffer(data[4:], dtype=np.uint8)
        self.state = torch.tensor(self.state).float().unsqueeze(0)
        policy, value, understanding = self.model(self.state)
        probabilities = torch.nn.functional.softmax(policy, dim=1)
        top_k_probs, top_k_indices = torch.topk(probabilities, TOPK)

        for i in range(TOPK):
            action_index = top_k_indices[0, i].item()
            if action_index in self.candidate:
                if action_index >= BOARD_POS:
                    self.action = action_index + MAP_POS_OFFSET
                else:
                    self.action = action_index
                print(self.action)
                return

        if self.candidate != None:
            self.action = random.choice(self.candidate)
            print("fall back:", self.action)
        else:
            print("This doesn't make any sense. Check it!")
            output(data)
            sys.exit(255)
        
