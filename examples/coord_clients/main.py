import random
import torch
import argparse
from random_agent import RandomAgent
from reinforcement_agent import RLAgent
from query_agent import QAgent
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathogen Agents for the coord server')
    parser.add_argument('--seed', type=str, help='Seed for the random number generator')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')
    parser.add_argument('-t', '--type', choices=['Random', 'Reinforcement', 'Query'], required=True,
                        help='Different types of Agent')
    parser.add_argument('-r', '--record', type=str, help='Record the game transaction in this file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Detailed information')
    parser.add_argument('-b', '--batch', type=int, help='Batch number', default = 1)

    # Only for ML agents
    parser.add_argument('-m', '--model', type=str, help='A trained pytorch model that provides (sub-)move to current game state', default='model.pth')
    parser.add_argument('-d', '--dataset', type=str, help='Output dataset to this file', default='dataset')

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(hash(args.seed))

    doctor_wins = 0
    for i in range(0, args.batch):
        if args.type == 'Random':
            a = RandomAgent(args)
        elif args.type == 'Reinforcement':
            a = RLAgent(args, i)
        elif args.type == 'Query':
            a = QAgent(args)

        # collect the result
        if args.side == 'Doctor' and a.result == b'Ix04':
            doctor_wins = doctor_wins + 1
        time.sleep(1)

    if args.side == 'Doctor':
        print(f"{doctor_wins/args.batch*100:6.2f} %")

