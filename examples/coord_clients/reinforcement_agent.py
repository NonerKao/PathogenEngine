import random
import os 
import sys
from base import Agent
from utils import output
import numpy as np
import torch
from constant import *

S_BOARD_MAP = BOARD_DATA + MAP_DATA
S_FLOW = FLOW_DATA
S_TURN = TURN_DATA

BDL = [S_BOARD_MAP, S_BOARD_MAP*16, S_BOARD_MAP*8, S_BOARD_MAP*4, S_BOARD_MAP*2, TOTAL_POS]
FDL = [S_FLOW+TOTAL_POS, 200, TOTAL_POS]
TDL = [S_TURN+TOTAL_POS, 200, TOTAL_POS]
SL = [TOTAL_POS*3, S_BOARD_MAP*2, S_BOARD_MAP, S_BOARD_MAP//2, S_BOARD_MAP//5, TOTAL_POS]

class PathogenNet(torch.nn.Module):
    def __init__(self):
        super(PathogenNet, self).__init__()
        self.b_layers = torch.nn.ModuleList([torch.nn.Linear(BDL[i], BDL[i+1]) for i in range(len(BDL) - 1)])
        self.f_layers = torch.nn.ModuleList([torch.nn.Linear(FDL[i], FDL[i+1]) for i in range(len(FDL) - 1)])
        self.t_layers = torch.nn.ModuleList([torch.nn.Linear(TDL[i], TDL[i+1]) for i in range(len(TDL) - 1)])
        self.s_layers = torch.nn.ModuleList([torch.nn.Linear(SL[i], SL[i+1]) for i in range(len(SL) - 1)])
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p=0.06)
        self.dropout2 = torch.nn.Dropout(p=0.02)
        
    def forward(self, x):
        xb = x[:, :S_BOARD_MAP] 
        xf = x[:, S_BOARD_MAP:S_BOARD_MAP+S_FLOW]
        xt = x[:, S_BOARD_MAP+S_FLOW:S_BOARD_MAP+S_FLOW+S_TURN]

        for layer in self.b_layers:
            xb = self.dropout1(self.leaky_relu(layer(xb)))

        xf = torch.cat((xb, xf), dim=1)
        for layer in self.f_layers:
            xf = layer(xf)

        xt = torch.cat((xb, xt), dim=1)
        for layer in self.t_layers:
            xt = layer(xt)

        xs = torch.cat((xb, xf, xt), dim=1)
        for layer in self.s_layers:
            xs = self.dropout2(self.leaky_relu(layer(xs)))

        return torch.softmax(xs, dim=1)

def init_model(model_name):
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
        print("Loaded saved model.")
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Starting with a new model.")
    return model

def init_optimizer(model):
    learning_rate = 0.009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    return optimizer, loss_func

class RLAgent(Agent):
    def __init__(self, f):
        super().__init__(f)
        self.action = 255
        self.prev_state = None
        self.index = TOTAL_POS
        self.index_array = []
        self.skip_inference = False

        # initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        if torch.cuda.is_available():
            print('Use CUDA')
        else:
            print('Use CPU')

        self.model = init_model(f.model)
        self.optimizer, self.loss_func = init_optimizer(self.model)
        self.online_training = f.online_training
        
        while self.play():
            continue
        if self.record is not None:
            self.record.close()

    def analyze(self, data):
        if ord('E') == data[0]:
            # Unfortunately, the previous sub-move is illegal.
            # A minus reward included transaction goes here.
            self.index = self.index + 1
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05', b'Ix06'):
            # This won't be really sent back to the server because the code
            # indicates that we have nothing to do now. This client still set a
            # action for the sake of dataset structure.
            # Ix00 and Ix02: our turn ends with the previous (sub-)move
            # Ix04: we won!
            # Ix05: we lost...
            # Ix06: somehow, either we or the component lost the connection
            self.action = 255
            return
        elif data[0:4] in (b'Ix01', b'Ix03'):
            # Either the previous sub-move succeeded or it is our turn now.
            # Reset the position candidate.
            self.index = 0
            self.skip_inference = False

        # Make next action
        if self.online_training:
            pass
        else:
            if not self.skip_inference:
                inputs = np.frombuffer(data[4:], dtype=np.uint8)
                inputs = torch.tensor(inputs).float().unsqueeze(0)
                _, self.index_array = torch.sort(self.model(inputs).squeeze(), descending=True)
                self.skip_inference = True
            print(self.index)
            self.action = self.index_array[self.index]
            if self.action > BOARD_POS:
                self.action = self.action + MAP_POS_OFFSET
        
