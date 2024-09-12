import random
import torch
import argparse
from reinforcement_simulator import RLSimAgent
from query_agent import QAgent
import time
import sys
import socket

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathogen Agents for the coord server')
    parser.add_argument('--seed', type=str, help='Seed for the random number generator')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')
    parser.add_argument('-t', '--type', choices=['Random', 'ReinforcementSimulate', 'ReinforcementPlay', 'Query'], required=True,
                        help='Different types of Agent')
    parser.add_argument('-r', '--record', type=str, help='Record the game transaction in this file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Detailed information')
    parser.add_argument('-b', '--batch', type=int, help='Batch number', default = 1)

    # Only for ML agents
    parser.add_argument('-m', '--model', type=str, help='A trained pytorch model that provides (sub-)move to current game state', default='model.pth')
    parser.add_argument('-d', '--dataset', type=str, help='Output dataset to this file', default='/dev/null')
    parser.add_argument('--trial-unit', type=int, help='The simulation size', default=20)
    parser.add_argument('--delay-unit', type=int, help='How many steps are skipped before simulation', default=0)
    parser.add_argument('--device', type=str, help='Simulation on what device', default='cpu')

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(hash(args.seed))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 6241 if args.side == "Doctor" else 3698))
    s.setblocking(1)
    s.settimeout(None)

    doctor_wins = 0
    for i in range(0, args.batch):
        if args.type == 'Random':
            a = QAgent(args, s)
        elif args.type == 'ReinforcementSimulate':
            a = RLSimAgent(args, s, i)
        elif args.type == 'Query':
            a = QAgent(args, s)

        # collect the result
        if args.side == 'Doctor' and a.result == b'Ix04':
            doctor_wins = doctor_wins + 1

    if args.side == 'Doctor':
        print(f"{doctor_wins/args.batch*100:6.2f} %", file=sys.stderr)

