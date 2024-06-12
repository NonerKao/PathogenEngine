import random
import os 
import sys
from base import Agent
from utils import output
import numpy as np
import torch
from constant import *

####
#       The hyperparameters
#
TRAINING_BATCH_UNIT = 100
TRAINING_EPOCH = 5000
TRAINING_EPOCH_UNIT = 250

ENV_SIZE = 5*6*6
MAP_SIZE = 25*1
INPUT_SIZE = ENV_SIZE + MAP_SIZE
OUTPUT_SIZE = 1
PAIR_SIZE = INPUT_SIZE + OUTPUT_SIZE

RES_SIZE = 12
RES_INPUT_SIZE = 64
NATURE_CHANNEL_SIZE = (9 + 2)

ACCURACY_THRESHOLD = 0.03
LEARNING_RATE = 0.0001

class PathogenResidualBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(SetupResidualBlock, self).__init__()
        # Convolution layer 1
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        
        # Convolution layer 2
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        # Store the original input for the residual connection
        residual = x
        
        # Convolution -> Batch Normalization -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        
        # Convolution -> Batch Normalization
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual
        out += residual
        
        # Final ReLU
        out = torch.nn.functional.relu(out)
        return out

class PathogenNet(torch.nn.Module):
    def __init__(self):
        super(PathogenNet, self).__init__()
        self.conv0 = torch.nn.Conv2d(NATURE_CHANNEL_SIZE, RES_INPUT_SIZE, kernel_size=3, stride=1, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(RES_INPUT_SIZE)
        self.relu0 = torch.nn.ReLU(inplace=True)

        # Using nn.ModuleList to add residual blocks
        self.resblocks0 = torch.nn.ModuleList([
            PathogenResidualBlock(RES_INPUT_SIZE) for _ in range(RES_SIZE)
        ])
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(RES_INPUT_SIZE, OUTPUT_SIZE)
        
    def forward(self, x):
        # Padding the game board and the map to get ready for a Nx6x7x7 tensor
        genv, gmap = x[:, 0: ENV_SIZE], x[:, ENV_SIZE: INPUT_SIZE]
        genv = torch.nn.functional.pad(genv.reshape(-1, 9, 6, 6), (1, 0, 1, 0), mode='constant', value=-1.0)
        gmap = torch.nn.functional.pad(gmap.reshape(-1, 2, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        x = torch.cat((genv, gmap), dim=1)

        # The init convolution part
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        # The residual part 0
        for block in self.resblocks0:
            x = block(x)

        # The output part
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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
    def __init__(self, f):
        super().__init__(f)
        self.action = 255
        self.all_transitions = torch.tensor([])
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
            else:
                self.valid_submove = self.valid_submove + 1
            self.move = self.move + 1
            self.submove = self.submove + 1
            temp = self.all_transitions[-self.index:].clone().to(dtype=torch.float32)
            states, actions, rewards = temp[:, :S].clone().to(dtype=torch.float32), temp[:, S], temp[:, S+1]
            if data[0:4] in (b'Ix04'):
                self.win = True
            self.action = 255
            return
        else:
            # Either the previous sub-move succeeded or it is our turn now.
            # Reset the position candidate.
            if self.online_training:
                pass
            self.inference = True
            if data[0:4] in (b'Ix01'):
                self.submove = self.submove + 1
                self.valid_submove = self.valid_submove + 1
                # worth some positive reward?
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
        self.model.eval()
        self.actions = torch.multinomial(self.model(self.state).squeeze(), TOTAL_POS, replacement=False)
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
            self.index = 0
            self.action_orig = self.actions[self.index]
            if self.online_training:
                self.model.train()
        if self.action_orig >= BOARD_POS:
            self.action = self.action_orig + MAP_POS_OFFSET
        else:
            self.action = self.action_orig
        self.action_tensor = torch.tensor(self.action_orig * 1.0).unsqueeze(0)
        
