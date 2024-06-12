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

BDL = [S_BOARD_MAP, S_BOARD_MAP*60, S_BOARD_MAP*10, TOTAL_POS]
FDL = [S_FLOW+TOTAL_POS, 200, TOTAL_POS]
TDL = [S_TURN+TOTAL_POS, 200, TOTAL_POS]
SL = [TOTAL_POS*3, S_BOARD_MAP*60, S_BOARD_MAP*10, S_BOARD_MAP//2, TOTAL_POS]

TRAIN_BATCH0 = 2
TRAIN_BATCH1 = 0
TRAIN_BATCH2 = 1
TRAIN_BATCH3 = 0

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
    learning_rate = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def range_remap(input_tensor, min=0.001, max=0.999):
    min_value = input_tensor.min(dim = 0, keepdim=True).values
    max_value = input_tensor.max(dim = 0, keepdim=True).values
    range_vals = max_value - min_value + 1e-5
    input = (input_tensor - min_value)/range_vals * (max - min) + min
    if input.ndim == 0:
        input = input.unsqueeze(0)
    return input

def loss_func_bisect(input, rewards):
    input = range_remap(input)
    try:
        ret = torch.dot(rewards, (input - 0.5)/((input - 1.0)*input*input))
        assert not torch.isnan(ret).any()
    except RuntimeError as e:
        print('???: ', e)
    return ret

def loss_func_bisect2(input, rewards):
    if input.ndim == 0:
        input = input.unsqueeze(0)
    try:
        ret = torch.dot(rewards, 12.0 * 1.0/(input - 1.01)*(input + 0.99)+13)
        assert not torch.isnan(ret).any()
    except RuntimeError as e:
        print('???: ', e)
        ret = 0.0
    return ret

class RLAgent(Agent):
    torch.set_default_device(torch.device("cuda"))
    result = {}
    result[b'Ix01'] = torch.tensor(1.0).to(dtype=torch.float32).unsqueeze(0) # SUBMOVE
    result[b'Ix02'] = torch.tensor(2.0).to(dtype=torch.float32).unsqueeze(0) # MOVE
    result[b'Ix03'] = None
    result[b'Ix04'] = torch.tensor(10000.0).to(dtype=torch.float32).unsqueeze(0) # WIN
    result[b'Ix00'] = torch.tensor(-10.0).to(dtype=torch.float32).unsqueeze(0) # PASSIVE
    result[b'Ix05'] = torch.tensor(10000.0).to(dtype=torch.float32).unsqueeze(0) # LOSE
    result[b'Ix06'] = None
    result[b'Ex'] = torch.tensor(-1.0).to(dtype=torch.float32).unsqueeze(0) # ILLEGAL
    def __init__(self, f):
        super().__init__(f)
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
            states, actions, rewards = self.all_transitions[:, :S].clone().to(dtype=torch.float32), self.all_transitions[:, S], self.all_transitions[:, S+1]
            for _ in range(0, TRAIN_BATCH0):
                self.optimizer.zero_grad()
                preds = self.model(states)
                assert not torch.isnan(preds).any()
                probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
                print(probs)
                loss = loss_func_bisect2(probs, rewards)
                loss.backward()
                self.optimizer.step()
            print('BATCH0')
            print()

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
                if torch.abs(ret.max()) < 1e-8:
                    return ret / (ret.max() + 1e-8)
                return ret / ret.max()
            def fit(input_tensor, head=0.0, tail=1.0):
                front = input_tensor[0]
                rear = input_tensor[-1]
                range_vals = front - rear + 1e-5
                return (input_tensor - front)/range_vals * (tail - head) + head

            states, actions, rewards = self.positive_transitions[:, :S], self.positive_transitions[:, S], self.positive_transitions[:, S+1]
            normalized_rewards = discount_and_normalize(rewards.flip(0).cumsum(dim=0).flip(0))
            for _ in range(0, TRAIN_BATCH1):
                self.optimizer.zero_grad()
                preds = self.model(states)
                assert not torch.isnan(preds).any()
                probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
                print(probs.shape)
                print(probs)
                if probs.ndim == 0:
                    probs = probs.unsqueeze(0)
                # Let's learn it this way after it gets a better understanding of the rule
                #if self.win:
                #    normalized_rewards = fit(normalized_rewards, head=2.0, tail=3.0)
                #    normalized_probs = normalize(probs, min=2.0, max=3.0)
                #else:
                #    normalized_rewards = fit(normalized_rewards, head=2.0, tail=1.0)
                #    normalized_probs = normalize(probs, min=1.0, max=2.0)
                def loss_func_negative_likelihood(input, rewards):
                    try:
                        ret = -1.0 * torch.dot(rewards, torch.log(input + 0.001))
                        try:
                            assert not torch.isnan(ret).any()
                        except AssertionError:
                            assert not torch.isnan(input).any()
                            assert not torch.isnan(rewards).any()
                            print(torch.log(input + 0.001))
                            assert not torch.isnan(torch.log(input + 0.001)).any()
                    except RuntimeError:
                        print(rewards.shape)
                        print(input.shape)
                        ret = 0.0
                    return ret
                loss = loss_func_negative_likelihood(probs, normalized_rewards)
                loss.backward(retain_graph=True)
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
            self.index = self.index + 1
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05', b'Ix06'):
            # This won't be really sent back to the server because the code
            # indicates that we have nothing to do now. This client still set a
            # action for the sake of dataset structure.
            # Ix00 and Ix02: our turn ends with the previous (sub-)move
            # Ix04: we won!
            # Ix05: we lost...
            # Ix06: somehow, either we or the component lost the connection
            if data[0:4] not in (b'Ix00'):
                self.positive_transitions = torch.cat((self.positive_transitions, temp), dim=0)
            else:
                self.valid_submove = self.valid_submove + 1
            self.move = self.move + 1
            self.submove = self.submove + 1
            def discount_and_normalize2(rewards, gamma=0.95):
                # Now we apply the multinomial method, so the front are more to blame
                # but the final one should remain
                coef = torch.pow(gamma, torch.arange((len(rewards))*1.0))
                coef[-1] = 1.0
                ret = coef * rewards
                if torch.abs(ret.max()) < 1e-8:
                    return ret / (ret.max() + 1e-8)
                return ret / ret.max()
            temp = self.all_transitions[-self.index:].clone().to(dtype=torch.float32)
            states, actions, rewards = temp[:, :S].clone().to(dtype=torch.float32), temp[:, S], temp[:, S+1]
            for _ in range(0, TRAIN_BATCH2):
                self.optimizer.zero_grad()
                preds = self.model(states)
                assert not torch.isnan(preds).any()
                probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
                print(probs)
                loss = loss_func_bisect2(probs, rewards)
                loss.backward(retain_graph=True)
                self.optimizer.step()
            print('BATCH2')
            if data[0:4] in (b'Ix04'):
                self.win = True
            self.action = 255
            return
        else:
            # Either the previous sub-move succeeded or it is our turn now.
            # Reset the position candidate.
            if self.online_training:
                self.positive_transitions = torch.cat((self.positive_transitions, temp), dim=0)
            self.inference = True
            if data[0:4] in (b'Ix01'):
                self.submove = self.submove + 1
                self.valid_submove = self.valid_submove + 1
                # worth some positive reward?
                def discount_and_normalize2(rewards, gamma=0.95):
                    # Now we apply the multinomial method, so the front are more to blame
                    # but the final one should remain
                    coef = torch.pow(gamma, torch.arange((len(rewards))*1.0))
                    coef[-1] = 1.0
                    ret = coef * rewards
                    if torch.abs(ret.max()) < 1e-8:
                        return ret / (ret.max() + 1e-8)
                    return ret / ret.max()
                temp = self.all_transitions[-self.index:].clone().to(dtype=torch.float32)
                states, actions, rewards = temp[:, :S].clone().to(dtype=torch.float32), temp[:, S], temp[:, S+1]
                #rewards = discount_and_normalize2(rewards)
                for _ in range(0, TRAIN_BATCH3):
                    self.optimizer.zero_grad()
                    preds = self.model(states)
                    assert not torch.isnan(preds).any()
                    probs = preds.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze().to(self.device)
                    print(probs)
                    loss = loss_func_bisect(probs, rewards)
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    pass
            elif data[0:4] in (b'Ix03'):
                # start of a new move, nothing to be done here
                pass

        # Make next action
        self.state = np.frombuffer(data[4:], dtype=np.uint8)
        self.state = torch.tensor(self.state).float().unsqueeze(0)
        if not self.inference:
            # explore
            if self.index < TOTAL_POS:
                self.action_orig = self.actions[self.index]
            else:
                print("This doesn't make any sense. Check it!")
                output(data)
                sys.exit(255)
        else:
            # exploit
            self.inference = False
            self.model.eval()
            self.actions = torch.multinomial(self.model(self.state).squeeze(), TOTAL_POS, replacement=False)
            self.index = 0
            self.action_orig = self.actions[self.index]
            if self.online_training:
                self.model.train()
        if self.action_orig >= BOARD_POS:
            self.action = self.action_orig + MAP_POS_OFFSET
        else:
            self.action = self.action_orig
        self.action_tensor = torch.tensor(self.action_orig * 1.0).unsqueeze(0)
        
