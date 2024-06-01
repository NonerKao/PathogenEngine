import numpy as np
import torch
import math
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
    
TRAINING_BATCH_UNIT = 50
TRAINING_EPOCH = 5000

ENV_SIZE = 5*6*6
MAP_SIZE = 25*1
INPUT_SIZE = ENV_SIZE + MAP_SIZE
OUTPUT_SIZE = 1
PAIR_SIZE = INPUT_SIZE + OUTPUT_SIZE

RES_SIZE = 12
RES_INPUT_SIZE = 64
NATURE_CHANNEL_SIZE = 6

ACCURACY_THRESHOLD = 0.03
LEARNING_RATE = 0.0001

class ScaledLoss(torch.nn.Module):
    def __init__(self, scale_factor=10000.0):
        super(ScaledLoss, self).__init__()
        self.scale_factor = scale_factor
        self.loss = torch.nn.MSELoss()

    def forward(self, input, target):
        loss = self.loss(input, target)
        scaled_loss = loss * self.scale_factor
        return scaled_loss

def init_optimizer(model):
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # for i, block in enumerate(model.resblocks):
    #    if i in range(0, 8):
    #        for param in block.parameters():
    #            param.requires_grad = False
    
    black_residuals = []
    white_residuals = []
    for i, block in enumerate(model.resblocks):
       if i%2 == 0:
           for param in block.parameters():
               black_residuals.append(param)
       else:
           for param in block.parameters():
               white_residuals.append(param)
    optimizer = torch.optim.Adam([
        {'params': black_residuals, 'lr': LEARNING_RATE/2},
        {'params': white_residuals, 'lr': LEARNING_RATE/3*2},
        {'params': [model.fc.bias, model.fc.weight], 'lr': LEARNING_RATE}  
    ], lr=LEARNING_RATE) 
    loss_func = ScaledLoss()
    return optimizer, loss_func

def init_dev():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_device(device)
        print("OK!")
        return device
    else:
        print("...")
        return None

class SetupResidualBlock(torch.nn.Module):
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

class SetupPredictorNet(torch.nn.Module):
    def __init__(self):
        super(SetupPredictorNet, self).__init__()
        self.conv0 = torch.nn.Conv2d(NATURE_CHANNEL_SIZE, RES_INPUT_SIZE, kernel_size=3, stride=1, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(RES_INPUT_SIZE)
        self.relu0 = torch.nn.ReLU(inplace=True)

        # Using nn.ModuleList to add residual blocks
        self.resblocks = torch.nn.ModuleList([
            SetupResidualBlock(RES_INPUT_SIZE) for _ in range(RES_SIZE)
        ])
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(RES_INPUT_SIZE, OUTPUT_SIZE)
        
    def forward(self, x):
        # Padding the game board and the map to get ready for a Nx6x7x7 tensor
        genv, gmap = x[:, 0: ENV_SIZE], x[:, ENV_SIZE: INPUT_SIZE]
        genv = torch.nn.functional.pad(genv.reshape(-1, 5, 6, 6), (1, 0, 1, 0))
        gmap = torch.nn.functional.pad(gmap.reshape(-1, 1, 5, 5), (1, 1, 1, 1))
        x = torch.cat((genv, gmap), dim=1)

        # The init convolution part
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        # The residual part
        for block in self.resblocks:
            x = block(x)

        # The output part
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def init_model(model_name):
    model = SetupPredictorNet()
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
        print("Loaded saved model.")
    else:
        # Start with a newly initialized model
        print("Starting with a new model.")
    return model

class SetupDataset(Dataset):
    def __init__(self, file_path, device):
        self.file = open(file_path, 'rb')
        self.pair_size = PAIR_SIZE*4
        self.device = device

    def __len__(self):
        return (os.fstat(self.file.fileno()).st_size) // self.pair_size

    def __getitem__(self, idx):
        start = idx * self.pair_size
        self.file.seek(start)
        data = self.file.read(self.pair_size)
        input_data = data[0:INPUT_SIZE*4]
        output_data = data[INPUT_SIZE*4:]
        input_data = torch.from_numpy(np.frombuffer(input_data, dtype=np.float32).copy()).to(self.device)
        output_data = torch.from_numpy(np.frombuffer(output_data, dtype=np.float32).copy()).to(self.device)

        return input_data, output_data

    def __del__(self):
        self.file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver for train/validate setup predictor for a Pathogen game')
    parser.add_argument('-t', '--train', type=str, help='train with the dataset', default='/dev/null')
    parser.add_argument('-v', '--validate', type=str, help='validate with the dataset', default='/dev/null')
    parser.add_argument('-m', '--model', type=str, help='an existing model', default='model.pth')
    parser.add_argument('-n', '--exp_name', type=str, help='the name of the recorded expiriment', default='runs/temp')

    args = parser.parse_args()
    device = init_dev()
    model = init_model(args.model)
    optimizer, loss_func = init_optimizer(model)

    writer = SummaryWriter(args.exp_name)
    if args.train != "/dev/null":
        t_dataset = SetupDataset(args.train, device)
        t_dataloader = DataLoader(t_dataset, batch_size=TRAINING_BATCH_UNIT, shuffle=True, generator=torch.Generator(device=device))
    v_dataset = SetupDataset(args.validate, device)
    v_dataloader = DataLoader(v_dataset, batch_size=TRAINING_BATCH_UNIT, shuffle=True, generator=torch.Generator(device=device))

    if args.train == "/dev/null":
        TRAINING_EPOCH = 1;
    max_pass_rate = 0.0
    for i in range(0, TRAINING_EPOCH):
        if args.train != "/dev/null":
            print('epoch: ', i)
            train_loss = 0.0
            for inputs, labels in t_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(t_dataloader.dataset)
            writer.add_scalar('Loss/train', train_loss, i)

        validate_loss = 0.0
        ok = 0
        for inputs, labels in v_dataloader:
            model.eval()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            if args.train != "/dev/null":
                model.train()
            diff = torch.abs(outputs-labels)
            for e in diff:
                if e <= ACCURACY_THRESHOLD:
                    ok = ok + 1
            validate_loss += loss.item() * inputs.size(0)
        ok /= len(v_dataloader.dataset)
        if max_pass_rate < ok:
            print("the real pass rate: ", ok, "; previous: ", max_pass_rate)
            max_pass_rate = ok
            torch.save(model, args.model)
        writer.add_scalar('Passrate/validate', ok, i)
        validate_loss /= len(v_dataloader.dataset)
        writer.add_scalar('Loss/validate', validate_loss, i)

    writer.close()
    os.sys.exit(0)
