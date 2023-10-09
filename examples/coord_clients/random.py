import argparse
from base import Agent

class Random(Agent):
    def __init__(self, f):
        super().__init__(f)

    def get_coord(self, data):
        self.action -= 1
        print(self.action)
        return self.action

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A Random Agent for Pathogen')
    parser.add_argument('-s', '--side', choices=['Doctor', 'Plague'], required=True,
                        help='Choose either "Docter" or "Plague"')
    
    args = parser.parse_args()
    d = Random(args.side)
    while True:
        d.play()
