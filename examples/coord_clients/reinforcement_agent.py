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
LEARNING_RATE = 0.001

RES_SIZE = 12
RES_INPUT_SIZE = 84 
FC_OUTPUT_SIZE = (1 + BOARD_POS + MAP_POS)
NATURE_CHANNEL_SIZE = (8 + 2 + 1 + 2 + 11)

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
        
        # Policy Head
        self.policy_conv = nn.Conv2d(RES_INPUT_SIZE, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 7 * 7, TOTAL_POS)

        # Value Head
        self.value_conv = nn.Conv2d(RES_INPUT_SIZE, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc = nn.Linear(1 * 7 * 7, 1)
        
        # Understanding Head
        # the prediction of error sub-move for all the positions
        # XXX: 4 is experimental magic
        self.understanding_conv = nn.Conv2d(RES_INPUT_SIZE, 4, kernel_size=1)
        self.understanding_bn = nn.BatchNorm2d(4)
        self.understanding_fc = nn.Linear(4 * 7 * 7, TOTAL_POS)
        
    def forward(self, x):
        # Padding the game board and the map to get ready for a Nx6x7x7 tensor
        genv, gmap, gturn, gfm, gfe = x[:, 0: BOARD_DATA], x[:, BOARD_DATA: E_MAP], x[:, E_MAP: E_TURN], x[:, E_TURN: E_FM], x[:, E_FM: S]
        # XXX: sort the server-side encoding later to eliminate all these permutations
        genv = genv.permute(0,3,1,2)
        gmap = gmap.permute(0,3,1,2)
        gturn = gturn.permute(0,3,1,2)
        gfm = gfm.permute(0,3,1,2)
        gfe = gfe.permute(0,3,1,2)
        genv = torch.nn.functional.pad(genv.reshape(-1, 8, 6, 6), (1, 0, 1, 0), mode='constant', value=-1.0)
        gmap = torch.nn.functional.pad(gmap.reshape(-1, 2, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        gturn = torch.nn.functional.pad(gmap.reshape(-1, 1, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        gfm = torch.nn.functional.pad(gmap.reshape(-1, 2, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        gfe = torch.nn.functional.pad(gmap.reshape(-1, 11, 6, 6), (1, 0, 1, 0), mode='constant', value=-1.0)
        x = torch.cat((genv, gmap, gturn, gfm, gfe), dim=1)

        # The init convolution part
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        # The residual part 0
        for block in self.resblocks0:
            x = block(x)

        # The output: policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = torch.nn.functional.relu(policy)
        policy = torch.flatten(policy, 1)
        policy = self.policy_fc(policy)
        policy = torch.nn.functional.softmax(policy, dim=1)

        # The output: value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = torch.nn.functional.relu(value)
        value = torch.flatten(value, 1)
        value = self.value_fc(value)
        value = torch.nn.functional.tanh(value)

        # The output: understanding head
        understanding = self.understanding_conv(x)
        understanding = self.understanding_bn(understanding)
        understanding = torch.nn.functional.relu(understanding)
        understanding = torch.flatten(understanding, 1)
        understanding = self.understanding_fc(understanding)
        understanding = torch.nn.functional.tanh(understanding)

        return policy, value, understanding

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
    learning_rate = LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

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
            # states, actions, rewards = self.all_transitions[:, :S].clone().to(dtype=torch.float32), self.all_transitions[:, S], self.all_transitions[:, S+1]

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
        
