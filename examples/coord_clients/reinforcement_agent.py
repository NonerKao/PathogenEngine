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
    torch.set_default_dtype(torch.float32)
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Warning: Starting with a new model.")
    return model

def init_optimizer(model):
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def loss_func(input, rewards):
    try:
        ret = torch.dot(rewards, 1/(input + 0.001)-0.999)
    except RuntimeError:
        print(rewards.shape)
        print(input.shape)
        print(torch.log(input + 0.01).shape)
        ret = 0.0
    return ret

class RLAgent(Agent):
    torch.set_default_device(torch.device("cuda"))
    result = {}
    result[b'Ix01'] = torch.tensor(np.log(3.5)).to(dtype=torch.float32).unsqueeze(0) # SUBMOVE
    result[b'Ix02'] = torch.tensor(np.log(4.0)).to(dtype=torch.float32).unsqueeze(0) # MOVE
    result[b'Ix03'] = None
    result[b'Ix04'] = torch.tensor(np.log(6.0)).to(dtype=torch.float32).unsqueeze(0) # WIN
    result[b'Ix00'] = torch.tensor(np.log(3.0)).to(dtype=torch.float32).unsqueeze(0) # PASSIVE
    result[b'Ix05'] = torch.tensor(np.log(5.0)).to(dtype=torch.float32).unsqueeze(0) # LOSE
    result[b'Ix06'] = None
    result[b'Ex'] = torch.tensor(np.log(2.0)).to(dtype=torch.float32).unsqueeze(0) # ILLEGAL
    def __init__(self, f):
        super().__init__(f)
        self.fixmap = list(range(TOTAL_POS))
        self.map = self.fixmap
        self.action = 255
        self.all_transitions = torch.tensor([])
        self.positive_transitions = torch.tensor([]) 
        self.win = False
        self.submove = 0
        self.valid_submove = 0
        self.move = 0
        self.score = 0
        self.inference = True

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
            def normalize(input_tensor, min=-1.0, max=1.0):
                min_value = input_tensor.min(dim = 0, keepdim=True).values
                max_value = input_tensor.max(dim = 0, keepdim=True).values
                range_vals = max_value - min_value + 1e-5
                return (input_tensor - min_value)/range_vals * (max - min) + min
            def bisect(input_tensor, alpha=0.5, beta=0.01):
                input_tensor[input_tensor >= 0] = alpha
                input_tensor[input_tensor < 0] = beta
                return input_tensor
            states, actions, rewards = self.all_transitions[:, :S].clone().to(dtype=torch.float32), self.all_transitions[:, S], self.all_transitions[:, S+1]
            self.optimizer.zero_grad()
            preds = self.model(states)
            assert not torch.isnan(preds).any()
            probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
            loss = loss_func(probs, rewards)
            loss.backward()
            self.optimizer.step()

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
            def fit(input_tensor, head=0.0, tail=1.0):
                front = input_tensor[0]
                rear = input_tensor[-1]
                range_vals = front - rear + 1e-5
                return (input_tensor - front)/range_vals * (tail - head) + head

            states, actions, rewards = self.positive_transitions[:, :S], self.positive_transitions[:, S], self.positive_transitions[:, S+1]
            normalized_rewards = discount_and_normalize(rewards.flip(0).cumsum(dim=0).flip(0))
            self.optimizer.zero_grad()
            preds = self.model(states)
            assert not torch.isnan(preds).any()
            probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
            # Let's learn it this way after it gets a better understanding of the rule
            #if self.win:
            #    normalized_rewards = fit(normalized_rewards, head=2.0, tail=3.0)
            #    normalized_probs = normalize(probs, min=2.0, max=3.0)
            #else:
            #    normalized_rewards = fit(normalized_rewards, head=2.0, tail=1.0)
            #    normalized_probs = normalize(probs, min=1.0, max=2.0)
            loss = loss_func(probs, normalized_rewards)
            loss.backward()
            self.optimizer.step()
            torch.save(self.model, f.model)
        if self.win:
            print("W ", end='')
        else:
            print("L ", end='')
        print(f"{self.score:10.2}", '/', f"{self.move: 3}", '/', f"{self.valid_submove: 4}", '/', f"{self.submove: 6}", '    ', f"{self.valid_submove/self.submove*100:7.2}", '%')

    def analyze(self, data):
        temp = torch.tensor([])
        try:
            r = RLAgent.result[data[:4]]
        except KeyError:
            r = RLAgent.result[b'Ex']
        if r is not None:
            if self.online_training:
                temp = torch.cat((self.state.squeeze(), (self.action_tensor).clone().detach().requires_grad_(True), r), dim=0).unsqueeze(0)
                self.all_transitions = torch.cat((self.all_transitions, temp), dim=0)
            self.score = self.score + r.item()

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
            self.submove = self.submove + 1
            self.valid_submove = self.valid_submove + 1
            self.move = self.move + 1
            if data[0:4] in (b'Ix04'):
                self.win = True
#            if self.online_training and data[0:4] in (b'Ix00', b'Ix02'):
#                temp = self.all_transitions[-self.submove:].clone().to(dtype=torch.float32)
#                states, actions = temp[:, :S], temp[:, S]
#                self.optimizer.zero_grad()
#                preds = self.model(states)
#                print(self.submove)
#                print(temp.shape)
#                assert not torch.isnan(preds).any()
#                probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
#                print(preds.shape)
#                print(probs.shape)
#                rewards = (torch.arange(self.submove) * 1.0).flip(0)
#                rewards = torch.pow(0.7, rewards)
#                loss = loss_func(probs, rewards)
#                loss.backward()
#                self.optimizer.step()
            self.action = 255
            return
        else:
            # Either the previous sub-move succeeded or it is our turn now.
            # Reset the position candidate.
            if self.online_training:
                self.positive_transitions = torch.cat((self.positive_transitions, temp), dim=0)
            self.map = self.fixmap.copy()
            self.inference = True
            if data[0:4] in (b'Ix01'):
                self.submove = self.submove + 1
                self.valid_submove = self.valid_submove + 1
                # worth some positive reward?
                pass
            elif data[0:4] in (b'Ix03'):
                # start of a new move, nothing to be done here
                pass

        # Make next action
        EPSILON = 0.99 - 0.02 * self.move
        if EPSILON <= 0.01:
            EPSILON = 0.01
        self.state = np.frombuffer(data[4:], dtype=np.uint8)
        self.state = torch.tensor(self.state).float().unsqueeze(0)
        if not self.inference or random.random() < EPSILON and self.online_training:
            # explore
            if len(self.map) != 0:
                self.action_orig = random.choice(self.map)
            else:
                print("This doesn't make any sense. Check it!")
                output(data)
                sys.exit(255)
        else:
            # exploit
            self.inference = False
            self.model.eval()
            self.action_orig = torch.argmax(self.model(self.state).squeeze()).item()
            if self.online_training:
                self.model.train()
            if self.action_orig not in self.map:
                self.action_orig = random.choice(self.map)
        if self.action_orig >= BOARD_POS:
            self.action = self.action_orig + MAP_POS_OFFSET
        else:
            self.action = self.action_orig
        self.action_tensor = torch.tensor(self.action_orig * 1.0).unsqueeze(0)
        
