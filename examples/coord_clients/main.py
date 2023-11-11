import random
import argparse
from random_agent import RandomAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathogen Agents for the coord server')
    parser.add_argument('--seed', type=str, help='Seed for the random number generator')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')
    parser.add_argument('-t', '--type', choices=['Random'], required=True,
                        help='Different types of Agent')
    parser.add_argument('-r', '--record', type=str, help='Record the game transaction in this file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Detailed information')

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if args.type == 'Random':
        RandomAgent(args)
