import random
import os 
import sys
from base import Agent
from utils import output
import numpy as np
import torch
from constant import *
from reinforcement_network import *

TOPK = 3

QUERY = 255
SAVE = 254
RETURN = 253
CLEAR = 252

TRIAL_UNIT = 10
DELAY_UNIT = 0

def init_model(model_name):
    torch.set_default_dtype(torch.float32)
    if os.path.exists(model_name):
        # Load the model state
        model = torch.load(model_name)
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Warning: Starting with a new model.")
        torch.save(model, model_name)
    return model

class Node():
    def __init__(self, state, parent, is_me):
        self.action = None
        self.state = state
        self.policy = None
        self.w = 0
        self.n = 0
        self.parent = parent
        self.child_nodes = {}
        self.is_me = is_me

    def update(self, w):
        coef = 1.0 if self.is_me else -1.0
        self.w += w * coef
        self.n += 1
        # print("=== this node ===")
        # print("action: ", self.action, "; state: ", self.state[:4] if self.state else None)
        # print("next: ", self.child_nodes.keys())
        # print("w: ", self.w, self.is_me)
        # print("n: ", self.n)
        # print("=== backtrack ===")
        if self.parent:
            self.parent.update(w)
        # else:
            # print("=================")
            # print("=== update end  =")
            # print("=================")
        return

class RLAgent(Agent):
    torch.set_default_device(torch.device("cuda"))
    def __init__(self, f):
        super().__init__(f)
        self.action = 255
        self.all_transitions = torch.tensor([])

        ### MCTS stuff
        # Most of the time this is True.
        # Only when we finish a bunch of MCTS simulations, 
        # we make a real play by revert this flag.
        self.simulation = False
        # when num_trials becomes 0, simulation goes from True to False;
        # when simulation goes from False to True, reset num_trials;
        # decrease this value each time a run of MCTS is finished
        self.num_trials = TRIAL_UNIT
        # I want it to "feel" the end game condition first.
        # the average steps of a game is around 30.
        self.delay = DELAY_UNIT
        # We are not like Go, where it is clear and definite who plays every action.
        self.is_me = True
        # The nodes
        self.root = None
        self.current_node = None

        # initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        if not torch.cuda.is_available():
            print('Warning: Use CPU')

        # We will only use this model for inference/simulation, at this phase
        self.model = init_model(f.model)
        self.model.eval()

        while self.play():
            continue
        if self.record is not None:
            self.record.close()
            # states, actions, rewards = self.all_transitions[:, :S].clone().to(dtype=torch.float32), self.all_transitions[:, S], self.all_transitions[:, S+1]

    def send_special(self, action):
        self.s.sendall(bytes([action]))
        self.num_candidate = int.from_bytes(self.s.recv(1), byteorder='big')
        self.candidate = self.s.recv(self.num_candidate)
        code = self.s.recv(CODE_DATA)
        while code not in (b'Wx00'):
            print("Unexpected query results:", self.num_candidate, "; candidates:", self.candidate);
            sys.exit(255)

    # result value:
    #    -1: lose
    #     0: win
    #     1: not sure yet, need inference
    def update(self, result):
        coef = 1.0 if self.current_node.is_me else -1.0
        w = 0
        if result == 1:
            state = np.frombuffer(self.current_node.state[4:], dtype=np.uint8)
            state = torch.tensor(state).float().unsqueeze(0)
            _, w, _ = self.model(state)
        else:
            w = result
        w = float(w) * coef
        self.current_node.update(w)
        self.current_node = self.root
        return

    def analyze(self, data):
        if ord('E') == data[0]:
            print("This doesn't make any sense. Check it!")
            sys.exit(255)
        elif data[0:4] in (b'Ix00', b'Ix02', b'Ix04', b'Ix05', b'Ix06'):
            # This won't be really sent back to the server because the code
            # indicates that we have nothing to do now. This client still set a
            # action for the sake of dataset structure.
            # Ix00 and Ix02: our turn ends with the previous (sub-)move
            # Ix04: we won!
            # Ix05: we lost...
            # Ix06: somehow, either we or the component lost the connection
            self.action = 255

            # [RL]
            # Since this move is finished, we decrease the delay value by 1,
            # if the delay was not yet zero'ed.
            # Otherwise: for Ix04/Ix05, they show that we are not in
            # a simulation and the game just ends;
            # clear nodes for normal closers Ix00/Ix02 when it is a real move.
            if self.delay > 0:
                self.delay = self.delay - 1
            if not self.simulation:
                self.root = None
                self.current_node = None
            return
        elif data[0:4] in (b'Ix01', b'Ix03', b'Ix07', b'Ix08'):
            if data[0:4] in (b'Ix07'):
                self.is_me = False
            elif data[0:4] in (b'Ix03', b'Ix08'):
                self.is_me = True
            else:
                assert self.current_node.parent != None, "An Ix01 node is supposed to have a parent"
                self.is_me = self.current_node.parent.is_me
            if self.root is not None and self.root.is_me is None:
                self.root.is_me = self.is_me
            if self.current_node is not None and self.current_node.is_me is None:
                self.current_node.is_me = self.is_me
            # [RL] starters and intermediate indicators
            # These events and special actions are not really for RL, but more
            # to make sure that we know
            # 1. What are the available moves. We did this back in the Query agent.
            # 2. When to save the status.
            # 3. When we run out of trials, actively CLEAR
            if self.delay > 0:
                # [RL]
                # The delay has not yet zero'ed, so we should just keep playing.
                self.action = QUERY
            elif not self.simulation:
                # [RL]
                # The situation diverges.
                # Ix03: We have a fresh new start. Make a new root now.
                # Ix07: A simulated opponent's start.
                # Ix08: A simulated start.
                # Ix01: A submove is done.
                if data[0:4] in (b'Ix03') or not self.root:
                    self.root = Node(data, None, self.is_me)
                    self.current_node = self.root
                elif not self.root.state:
                    self.root.state = data
                    self.current_node = self.root

                if self.current_node.child_nodes and len(self.current_node.child_nodes) <= 1:
                    # no point to SAVE such a root
                    self.action = QUERY
                else:
                    # Ideally most of the time this is trying simulated moves, so if 
                    # it is not in a simulation, just enable it.
                    # We can see this SAVE as a special variant of a RETURN to the
                    # root state because this one has no child currently.
                    self.root.parent = None
                    self.action = SAVE
                    self.num_trials = TRIAL_UNIT
                    self.simulation = True
            else: # self.simulation
                # [RL]
                # Do a simulation then. What nodes are we exploring? Does it have
                # child states?
                if not self.current_node.state:
                    self.current_node.state = data
                else:
                    assert self.current_node.state == data, "state not retrieved correctly"
                if self.num_trials <= 0:
                    # [RL]
                    # Now the simulation is done.
                    self.simulation = False
                    self.action = CLEAR
                    self.update(1)
                else:
                    self.action = QUERY

            self.send_special(self.action)
            if self.simulation and not self.current_node.child_nodes:
                for action in self.candidate:
                    self.current_node.child_nodes[action] = Node(None, self.current_node, None)
                # why would we backtracking for this single option?
                if len(self.candidate) > 1:
                    self.num_trials = self.num_trials - 1
                    self.action = RETURN
                    self.send_special(self.action)
                    self.update(1)
        elif data[0:4] in (b'Ix09', b'Ix0a'):
            self.num_trials = self.num_trials - 1
            if self.num_trials <= 0:
                self.action = CLEAR
                self.simulation = False
            else:
                self.action = RETURN
            self.update(-1 if data[0:4] == b'Ix09' else 0)
            self.send_special(self.action)
        else:
            print("What is this?")
            panic()

        self.action = None
        if not self.simulation:
            # Make next action
            self.state = np.frombuffer(self.current_node.state[4:], dtype=np.uint8)
            self.state = torch.tensor(self.state).float().unsqueeze(0)
            policy, value, valid = self.model(self.state)
            probabilities = torch.nn.functional.softmax(policy, dim=1)
            top_k_probs, top_k_indices = torch.topk(probabilities, TOPK)

            for i in range(TOPK):
                action_index = top_k_indices[0, i].item()
                if action_index in self.candidate:
                    if action_index >= BOARD_POS:
                        self.action = action_index + MAP_POS_OFFSET
                    else:
                        self.action = action_index

            if self.action is None:
                if self.candidate != None:
                    self.action = random.choice(self.candidate)
                else:
                    print("This doesn't make any sense. Check it!")
                    output(data)
                    sys.exit(255)

            # [RL]
            # not simulating? we just made a real action. Next one will
            # start with a new root
            if self.delay <= 0:
                self.current_node = self.root.child_nodes[self.action]
                self.root = self.current_node

        else:
            # [RL]
            # XXX: Apply the MCTS result and the boltzman distribution
            if self.candidate != None:
                self.action = random.choice(self.candidate)
            else:
                print("This doesn't make any sense. Check it!")
                output(data)
                sys.exit(255)

            self.current_node = self.current_node.child_nodes[self.action]

        self.current_node.action = self.action
