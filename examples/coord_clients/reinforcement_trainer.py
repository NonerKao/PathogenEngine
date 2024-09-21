import numpy as np
import torch
import math
import sys
import os
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from reinforcement_network import *
from sklearn.model_selection import KFold

TRAINING_BATCH_UNIT = 15
TRAINING_INNER_EPOCH = 1
TRAINING_OUTER_EPOCH = 6

LEARNING_RATE = 0.0005
KFOLD = 4

def init_optimizer(model):
    # To apply the LR globally
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # To stop update some part in the model
    # for i, block in enumerate(model.resblocks):
    #    if i in range(0, 8):
    #        for param in block.parameters():
    #            param.requires_grad = False

    # Zebra pattern
    # black_residuals = []
    # white_residuals = []
    # for i, block in enumerate(model.resblocks):
    #   if i%2 == 0:
    #       for param in block.parameters():
    #           black_residuals.append(param)
    #   else:
    #       for param in block.parameters():
    #           white_residuals.append(param)
    # optimizer = torch.optim.Adam([
    #     {'params': black_residuals, 'lr': LEARNING_RATE/2},
    #     {'params': white_residuals, 'lr': LEARNING_RATE/3*2},
    #     {'params': [model.fc.bias, model.fc.weight], 'lr': LEARNING_RATE}
    # ], lr=LEARNING_RATE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return optimizer

def init_dev():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_device(device)
        print("OK!")
        return device
    else:
        print("...")
        return None

def init_model(model_name):
    torch.set_default_dtype(torch.float32)
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
        print("Loaded saved model.")
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Warning: Starting with a new model.")
        torch.save(model, model_name)
    return model

class SimulationDataset(Dataset):
    def __init__(self, file_path, device):
        self.file = open(file_path, 'rb')
        self.device = device

    def __len__(self):
        return (os.fstat(self.file.fileno()).st_size) // DATASET_UNIT

    def __getitem__(self, idx):
        start = idx * DATASET_UNIT
        self.file.seek(start)
        data = self.file.read(DATASET_UNIT)
        state = data[0:D_STATE]
        policy = data[D_STATE:D_STATE+D_POLICY]
        valid = data[D_STATE+D_POLICY:D_STATE+D_POLICY+D_VALID]
        value = data[D_STATE+D_POLICY+D_VALID:D_STATE+D_POLICY+D_VALID+D_VALUE]
        input_data = torch.from_numpy(np.frombuffer(state, dtype=np.float32).copy()).to(self.device)
        output_head1 = torch.from_numpy(np.frombuffer(policy, dtype=np.float32).copy()).to(self.device)
        output_head2 = torch.from_numpy(np.frombuffer(valid, dtype=np.float32).copy()).to(self.device)
        output_head3 = torch.from_numpy(np.frombuffer(value, dtype=np.float32).copy()).to(self.device)

        return input_data, output_head1, output_head2, output_head3

    def __del__(self):
        self.file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver for train/validate setup predictor for a Pathogen game')
    parser.add_argument('-d', '--dataset', type=str, help='the dataset', default='/dev/null')
    parser.add_argument('-t', '--test-dataset', type=str, help='dataset for testing', default='/dev/null')
    parser.add_argument('-m', '--model', type=str, help='an existing model', default='model.pth')
    parser.add_argument('-n', '--exp_name', type=str, help='the name of the recorded expiriment', default='runs/temp')

    args = parser.parse_args()
    device = init_dev()

    writer = SummaryWriter(args.exp_name)
    model = init_model(args.model)
    optimizer = init_optimizer(model)

    kfold = KFold(n_splits=KFOLD, shuffle=True)
    simulation_dataset = SimulationDataset(args.dataset, device)
    test_dataset = SimulationDataset(args.test_dataset, device)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=TRAINING_BATCH_UNIT,
        shuffle=False,
        generator=torch.Generator(device=device)
    )

    # Since I am using sum reduction for the valid check, the weight is thus
    # implicitly larger than other ones. I think that makes sense.
    policy_loss_func = torch.nn.CrossEntropyLoss()
    valid_loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
    value_loss_func = torch.nn.MSELoss()

    # cross-validation
    i = 0
    for o_epoch in range(TRAINING_OUTER_EPOCH):
        for fold, (train_idx, val_idx) in enumerate(kfold.split(simulation_dataset)):

            # Sample elements from indices
            train_subset = Subset(simulation_dataset, train_idx)
            val_subset = Subset(simulation_dataset, val_idx)

            # Create DataLoaders
            train_dataloader = DataLoader(
                train_subset,
                batch_size=TRAINING_BATCH_UNIT,
                shuffle=True,
                generator=torch.Generator(device=device)
            )
            val_dataloader = DataLoader(
                val_subset,
                batch_size=TRAINING_BATCH_UNIT,
                shuffle=False,
                generator=torch.Generator(device=device)
            )

            for i_epoch in range(TRAINING_INNER_EPOCH):
                train_loss = [0.0]
                model.train()
                for state, policy, valid, value in train_dataloader:
                    value_pred = model(state)
                    value_loss = value_loss_func(value_pred, value)
                    train_loss[0] += value_loss.item() * state.size(0)
                    print(f'({o_epoch}, {fold}, {i_epoch})/({TRAINING_OUTER_EPOCH}, {KFOLD}, {TRAINING_INNER_EPOCH})')
                    sys.stdout.write("\033[F")
                    sys.stdout.flush()
                    optimizer.zero_grad()
                    value_loss.backward()
                    optimizer.step()

                train_loss = [x / len(train_dataloader.dataset) for x in train_loss]
                writer.add_scalar('Loss/train_value', train_loss[0], i)

                val_loss = [0.0]
                model.eval()
                with torch.no_grad():
                    for state, policy, valid, value in val_dataloader:
                        value_pred = model(state)
                        value_loss = value_loss_func(value_pred, value)
                        val_loss[0] += value_loss.item() * state.size(0)
                    val_loss = [x / len(val_dataloader.dataset) for x in val_loss]
                    writer.add_scalar('Loss/val_value', val_loss[0], i)

                test_loss = [0.0]
                model.eval()
                with torch.no_grad():
                    for state, policy, valid, value in test_dataloader:
                        value_pred = model(state)
                        value_loss = value_loss_func(value_pred, value)
                        test_loss[0] += value_loss.item() * state.size(0)
                    test_loss = [x / len(test_dataloader.dataset) for x in test_loss]
                    writer.add_scalar('Loss/test_value', test_loss[0], i)

                i += 1
                torch.save(model, args.model+'.'+str(i))

    writer.close()
    os.sys.exit(0)
