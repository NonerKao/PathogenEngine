import torch
import numpy as np
import sys

def init_misc():
    sys.path.append('..')
    import reinforcement_agent
    m = torch.load('../tmp10.pth')
    f = open('/opt/dataset/random/validating_200/doc.bin', 'rb')
    torch.set_default_device(torch.device("cuda"))
    return f, m

def evaluate(f, m):
    input_data = f.read(388)
    f.read(4)
    input_data = torch.from_numpy(np.frombuffer(input_data[:387], dtype=np.uint8).copy()).float().to(torch.device("cuda"))
    camp = torch.from_numpy(np.frombuffer(b'\x01\x00', dtype=np.uint8).copy()).float().to(torch.device("cuda"))
    input_data = torch.cat((input_data, camp), dim=0).unsqueeze(0).to(torch.device("cuda"))
    return m(input_data)
