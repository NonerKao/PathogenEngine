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
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Warning: Starting with a new model.")
    return model

def init_optimizer(model):
    learning_rate = 0.009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

class RLAgent(Agent):
    torch.set_default_device(torch.device("cuda"))
    result = {}
    result[b'Ix01'] = torch.tensor(1.0).unsqueeze(0) # SUBMOVE
    result[b'Ix02'] = torch.tensor(2.0).unsqueeze(0) # MOVE
    result[b'Ix03'] = None
    result[b'Ix04'] = torch.tensor(10.0).unsqueeze(0) # WIN
    result[b'Ix00'] = torch.tensor(-2.0).unsqueeze(0) # PASSIVE
    result[b'Ix05'] = torch.tensor(-4.0).unsqueeze(0) # LOSE
    result[b'Ix06'] = None
    result[b'Ex'] = torch.tensor(-8.0).unsqueeze(0) # ILLEGAL
    def __init__(self, f):
        super().__init__(f)
        self.fixmap = list(range(TOTAL_POS))
        self.map = self.fixmap
        self.action = 255
        self.all_transitions = torch.tensor([])
        self.positive_transitions = torch.tensor([]) 

        # initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        if not torch.cuda.is_available():
            print('Warning: Use CPU')

        self.model = init_model(f.model)
        self.optimizer = init_optimizer(self.model)
        self.online_training = f.online_training
        
        while self.play():
            continue
        if self.record is not None:
            self.record.close()
        if self.online_training:
            # sort out the two views of history
            # the first is the all history, including two negatice behaviors:
            # illegal (sub-)moves and passes. In this phase, I want each action
            # takes the feedback it deserves, independent from consecutive ones.
#            states, actions, rewards = self.all_transitions[:, :S], self.all_transitions[:, S], self.all_transitions[:, S+1]
#            def normalize(input_tensor, min=-1.0, max=1.0):
#                min_value = input_tensor.min(dim = 0, keepdim=True).values
#                max_value = input_tensor.max(dim = 0, keepdim=True).values
#                range_vals = max_value - min_value + 1e-5
#                return (input_tensor - min_value)/range_vals * (max - min) + min
#
#            normalized_rewards = normalize(rewards)
#            preds = self.model(states)
#            probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
#            if probs.dim() == 0:
#                pass
#            elif len(probs) >= 1:
#                print(probs[-1])
#            def loss_func1(input, rewards):
#                return -1.0 * torch.dot(torch.softmax(rewards, dim=0), input)
#            loss = loss_func1(probs, normalized_rewards)
#            self.optimizer.zero_grad()
#            loss.backward()
#            self.optimizer.step()

            # the second one counts only the valid (sub-)moves. since the
            # actions do contributes to the results, I want the discount credit
            # assignment here.
            def discount_and_normalize(rewards, gamma=0.95):
                # This one is a bit confusing.
                # based on "DRL in Action", derived from section 4.2.4 and 4.2.5,
                # the closer to the end, the less discount it should be. However,
                # in section 4.4.3 programm 4.6, the order is totally reveresed.
                # Here we follow the former.
                coef = torch.pow(gamma, torch.arange((len(rewards))*1.0)).flip(0)
                ret = coef * rewards
                return ret / ret.max()
#            states, actions, rewards = self.positive_transitions[:, :S], self.positive_transitions[:, S], self.positive_transitions[:, S+1]
#            normalized_rewards = discount_and_normalize(rewards.flip(0).cumsum(dim=0).flip(0))
#            preds = self.model(states)
#            probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
#            def loss_func2(input, rewards):
#                return -1.0 * torch.sum(rewards * torch.log(input))
#            loss = loss_func2(probs, normalized_rewards)
#            self.optimizer.zero_grad()
#            loss.backward()
#            self.optimizer.step()
            torch.save(self.model, f.model)
            print('total sub-moves: ', str(len(self.all_transitions)))
            print('valid ratio: ', str(len(self.positive_transitions)/len(self.all_transitions)))

    def analyze(self, data):
        temp = torch.tensor([])
        if self.online_training:
            try:
                r = RLAgent.result[data[:4]]
            except KeyError:
                r = RLAgent.result[b'Ex']
            if r is not None:
                temp = torch.cat((self.state.squeeze(), (self.action_tensor).clone().detach().requires_grad_(True), r), dim=0).unsqueeze(0)
                self.all_transitions = torch.cat((self.all_transitions, temp), dim=0)

        if ord('E') == data[0]:
            # Unfortunately, the previous sub-move is illegal.
            # A minus reward included transaction goes here.
            self.submove = self.submove + 1
            try:
                self.map.remove(self.action_orig)
            except ValueError:
                pass
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05', b'Ix06'):
            # This won't be really sent back to the server because the code
            # indicates that we have nothing to do now. This client still set a
            # action for the sake of dataset structure.
            # Ix00 and Ix02: our turn ends with the previous (sub-)move
            # Ix04: we won!
            # Ix05: we lost...
            # Ix06: somehow, either we or the component lost the connection
            self.positive_transitions = torch.cat((self.positive_transitions, temp), dim=0)
            if self.online_training:
                if data[0:4] in (b'Ix00', b'Ix02'):
                    self.submove = self.submove + 1
                    self.move = self.move + 1
                    temp = self.all_transitions[-self.submove:]
                    states, actions = temp[:, :S], temp[:, S]
                    preds = self.model(states)
                    probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
                    rewards = (torch.arange(self.submove) * 1.0).flip(0)
                    rewards = torch.pow(0.7, rewards)
                    assert not torch.isnan(probs).any()
                    assert not torch.isnan(rewards).any()
                    loss_func = torch.nn.MSELoss()
                    loss = loss_func(probs, rewards)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                elif data[0:4] in (b'Ix04'):
                    print()
                    print("win!")
                elif data[0:4] in (b'Ix05'):
                    print()
                    print("lose!")
            self.action = 255
            return
        else:
            # Either the previous sub-move succeeded or it is our turn now.
            # Reset the position candidate.
            self.positive_transitions = torch.cat((self.positive_transitions, temp), dim=0)
            self.map = self.fixmap.copy()
            if data[0:4] in (b'Ix01'):
                self.submove = 0
                self.move = self.move + 1
                # worth some positive reward?
                pass
            elif data[0:4] in (b'Ix03'):
                self.move = 0
                self.submove = 0
                # start of a new move, nothing to be done here
                pass

        # Make next action
        EPSILON = 0.8 - 0.001 * self.submove
        if EPSILON <= 0.0:
            EPSILON = 0.2
        self.state = np.frombuffer(data[4:], dtype=np.uint8)
        self.state = torch.tensor(self.state).float().unsqueeze(0)
        if random.random() < EPSILON:
            # explore
            if len(self.map) != 0:
                self.action_orig = random.choice(self.map)
            else:
                print("This doesn't make any sense. Check it!")
                output(data)
                sys.exit(255)
        else:
            # exploit
            self.action_orig = torch.argmax(self.model(self.state).squeeze()).item()
        if self.action_orig > BOARD_POS:
            self.action = self.action_orig + MAP_POS_OFFSET
        else:
            self.action = self.action_orig
        self.action_tensor = torch.tensor(self.action_orig * 1.0).unsqueeze(0)
        
