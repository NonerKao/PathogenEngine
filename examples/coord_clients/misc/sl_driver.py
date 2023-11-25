import numpy as np
import torch
import math
import os
import argparse
from torch.utils.data import Dataset, DataLoader

Q_SIZE = 374
FC_SIZE = 13
ACTION_SIZE = 1
STATUS_SIZE = 4

Q_DECODER_LAYERS=6
QDL = [Q_SIZE, Q_SIZE*16, Q_SIZE*8, Q_SIZE*4, Q_SIZE*2, 61]
FC_DECODER_LAYERS=2
FDL = [FC_SIZE, 61]
POS_DECODER_LAYERS=3
PDL = [1, 100, 61]
STRATEGY_LAYERS=5
SL = [122, Q_SIZE, Q_SIZE//2, Q_SIZE//5, 61]
T = 2

def init_dev():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_device(device)
        print("OK!")
        return device
    else:
        print("...")
        return None

def rescale(tensor, new_min, new_max):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized = (tensor - min_val) / (max_val - min_val)
    rescaled = normalized * (new_max - new_min) + new_min
    return rescaled

class PathogenDoctorNet(torch.nn.Module):
    def __init__(self):
        super(PathogenDoctorNet, self).__init__()
        self.q_layers = torch.nn.ModuleList([torch.nn.Linear(QDL[i], QDL[i+1]) for i in range(Q_DECODER_LAYERS - 1)])
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(FDL[i], FDL[i+1]) for i in range(FC_DECODER_LAYERS - 1)])
        self.pos_layers = torch.nn.ModuleList([torch.nn.Linear(PDL[i], PDL[i+1]) for i in range(POS_DECODER_LAYERS - 1)])
        self.s_layers = torch.nn.ModuleList([torch.nn.Linear(SL[i], SL[i+1]) for i in range(STRATEGY_LAYERS - 1)])
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p=0.05)
        self.dropout2 = torch.nn.Dropout(p=0.01)
        
    def forward(self, x):
        xq, xf, xp = x[:, :Q_SIZE], x[:, Q_SIZE:Q_SIZE+FC_SIZE], x[:, Q_SIZE+FC_SIZE]

        for layer in self.q_layers:
            xq = self.dropout1(self.leaky_relu(layer(xq)))
        for layer in self.fc_layers:
            xf = layer(xf)
        i = 0
        for layer in self.pos_layers:
            xp = layer(xp.reshape(len(xp),PDL[i]))
            i = i + 1

        legal = torch.mul(xf, xp)
        substep = torch.cat((xq, xf), dim=1)
        for layer in self.s_layers:
            substep = self.dropout2(self.leaky_relu(layer(substep)))

        return rescale(torch.sum(torch.mul(substep, legal), dim=1), 0.0, 1.0)

def init_model(model_name):
    model = PathogenDoctorNet()
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
        print("Loaded saved model.")
    else:
        # Start with a newly initialized model
        print("Starting with a new model.")
    return model

def init_optimizer():
    learning_rate = 0.009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    return optimizer, loss_func

class CustomPathogenDataset(Dataset):
    def __init__(self, file_path):
        self.file = open(file_path, 'rb')
        self.pair_size = Q_SIZE+FC_SIZE+ACTION_SIZE+STATUS_SIZE

    def __len__(self):
        return (os.fstat(self.file.fileno()).st_size - STATUS_SIZE)// self.pair_size

    def __getitem__(self, idx):
        start = STATUS_SIZE + idx * self.pair_size
        self.file.seek(start)
        data = self.file.read(self.pair_size)
        input_data = data[0:Q_SIZE+FC_SIZE+ACTION_SIZE]
        if data[-4] == ord('E'):
            output_data = b'\x00'
        else:
            output_data = b'\x01'
        input_data = torch.from_numpy(np.frombuffer(input_data, dtype=np.uint8).copy()).float().to(device)
        output_data = torch.from_numpy(np.frombuffer(output_data, dtype=np.uint8).copy()).float().to(device)

        return input_data, output_data

    def __del__(self):
        self.file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver for train/validate Pathogen Models')
    parser.add_argument('-t', '--train', action='store_true', help='train the model')
    parser.add_argument('-v', '--validate', action='store_true', help='validate the model')
    parser.add_argument('-m', '--model', type=str, help='an existing model', default='model.pth')
    parser.add_argument('-d', '--dataset', type=str, help='a directory that contains Pathogen Dataset', default='/opt/dataset/random/training_1K/doc.bin')

    args = parser.parse_args()
    device = init_dev()
    model = init_model(args.model)
    optimizer, loss_func = init_optimizer()
    
    TRAINING_BATCH_UNIT = 1000
    TRAINING_EPOCH = 3
    TRAINING_CHECKPOINT = 100

    dataset = CustomPathogenDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=TRAINING_BATCH_UNIT, shuffle=False)

    if args.train:
        for i in range(0, TRAINING_EPOCH):
            j = 0
            for inputs, labels in dataloader:
                print('epoch: ', i, '   batch: ', j)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
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
