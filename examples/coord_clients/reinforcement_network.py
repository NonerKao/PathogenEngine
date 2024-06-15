import numpy as np
import torch
from constant import *

RES_SIZE = 12
RES_INPUT_SIZE = 84 
FC_OUTPUT_SIZE = (1 + BOARD_POS + MAP_POS)
NATURE_CHANNEL_SIZE = (8 + 2 + 1 + 2 + 11)

class PathogenResidualBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(PathogenResidualBlock, self).__init__()
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
        self.policy_conv = torch.nn.Conv2d(RES_INPUT_SIZE, 2, kernel_size=1)
        self.policy_bn = torch.nn.BatchNorm2d(2)
        self.policy_fc = torch.nn.Linear(2 * 7 * 7, TOTAL_POS)

        # Value Head
        self.value_conv = torch.nn.Conv2d(RES_INPUT_SIZE, 1, kernel_size=1)
        self.value_bn = torch.nn.BatchNorm2d(1)
        self.value_fc = torch.nn.Linear(1 * 7 * 7, 1)
        
        # Understanding Head
        # the prediction of error sub-move for all the positions
        # XXX: 4 is experimental magic
        self.understanding_conv = torch.nn.Conv2d(RES_INPUT_SIZE, 4, kernel_size=1)
        self.understanding_bn = torch.nn.BatchNorm2d(4)
        self.understanding_fc = torch.nn.Linear(4 * 7 * 7, TOTAL_POS)
        
    def forward(self, x):
        # Padding the game board and the map to get ready for a Nx6x7x7 tensor
        genv, gmap, gturn, gfm, gfe = x[:, 0: BOARD_DATA], x[:, BOARD_DATA: E_MAP], x[:, E_MAP: E_TURN], x[:, E_TURN: E_FM], x[:, E_FM: S]
        # XXX: sort the server-side encoding later to eliminate all these permutations
        genv = genv.reshape(-1, 6, 6, 8).permute(0,3,1,2)
        gmap = gmap.reshape(-1, 5, 5, 2).permute(0,3,1,2)
        gturn = gturn.reshape(-1, 5, 5, 1).permute(0,3,1,2)
        gfm = gfm.reshape(-1, 5, 5, 2).permute(0,3,1,2)
        gfe = gfe.reshape(-1, 6, 6, 11).permute(0,3,1,2)
        genv = torch.nn.functional.pad(genv.reshape(-1, 8, 6, 6), (1, 0, 1, 0), mode='constant', value=-1.0)
        gmap = torch.nn.functional.pad(gmap.reshape(-1, 2, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        gturn = torch.nn.functional.pad(gturn.reshape(-1, 1, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        gfm = torch.nn.functional.pad(gfm.reshape(-1, 2, 5, 5), (1, 1, 1, 1), mode='constant', value=-1.0)
        gfe = torch.nn.functional.pad(gfe.reshape(-1, 11, 6, 6), (1, 0, 1, 0), mode='constant', value=-1.0)
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