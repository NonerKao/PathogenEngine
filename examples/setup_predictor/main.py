import numpy as np
import torch
import math
import os
import argparse
from torch.utils.data import Dataset, DataLoader
    
TRAINING_BATCH_UNIT = 10
TRAINING_EPOCH = 50
TRAINING_CHECKPOINT = 30

ENV_SIZE = 5*6*6
MAP_SIZE = 25*1
INPUT_SIZE = ENV_SIZE + MAP_SIZE
OUTPUT_SIZE = 3891
PAIR_SIZE = INPUT_SIZE + OUTPUT_SIZE

RES_SIZE = 12
RES_INPUT_SIZE = 64
NATURE_CHANNEL_SIZE = 6

def init_optimizer():
    learning_rate = 0.009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = ScaledL1Loss()
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

class ScaledL1Loss(torch.nn.Module):
    def __init__(self, scale_factor=10000.0):
        super(ScaledL1Loss, self).__init__()
        self.scale_factor = scale_factor
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input, target):
        loss = self.l1_loss(input, target)
        scaled_loss = loss * self.scale_factor
        return scaled_loss

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
        x = torch.nn.functional.softmax(x, dim=1)

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
    parser.add_argument('-t', '--train', action='store_true', help='train the model')
    parser.add_argument('-v', '--validate', action='store_true', help='validate the model')
    parser.add_argument('-m', '--model', type=str, help='an existing model', default='model.pth')
    parser.add_argument('-d', '--dataset', type=str, help='a directory that contains setup dataset', default='../setup_generator/1000s_100b/1000s_100b.bin')

    args = parser.parse_args()
    device = init_dev()
    model = init_model(args.model)
    optimizer, loss_func = init_optimizer()

    dataset = SetupDataset(args.dataset, device)
    dataloader = DataLoader(dataset, batch_size=TRAINING_BATCH_UNIT, shuffle=True, generator=torch.Generator(device=device))

    if args.train:
        for i in range(0, TRAINING_EPOCH):
            j = 0
            for inputs, labels in dataloader:
                print('epoch: ', i, '   batch: ', j)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                print(loss)
                loss.backward()
                optimizer.step()
                j = j + 1
                if j % TRAINING_CHECKPOINT == 0:
                    torch.save(model, args.model)
                    if args.validate:
                        model.eval()
                        outputs = model(inputs)
                        loss = loss_func(outputs, labels)
                        print(loss)
                        model.train()
        os.sys.exit(0)
    if args.validate:
        model.eval()
        torch.no_grad()
        error_count = 0
        valid_count = 0
        error_total = 0
        valid_total = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            for i in range(0, len(outputs)):
                if labels[i] < 0.5:
                    error_total = error_total + 1
                    if outputs[i] < 0.5:
                        error_count = error_count + 1
                if labels[i] >= 0.5:
                    valid_total = valid_total + 1
                    if outputs[i] >= 0.5:
                        valid_count = valid_count + 1
        print('accuracy: ', str((error_count+valid_count)/(error_total+valid_total)))
        print('error case accuracy: ', str(error_count/(error_total)))
        print('valid case accuracy: ', str(valid_count/(valid_total)))
