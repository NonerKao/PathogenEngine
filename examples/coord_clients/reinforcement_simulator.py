import random
import os
import sys
from base import Agent
from utils import output
import numpy as np
import torch
from constant import *
from reinforcement_network import *
from collections import Counter

TEMPERATURE = 2.0
SPICE = 2

def init_model(args):
    torch.set_default_dtype(torch.float32)
    if args.model == '/dev/null':
        model = NullModel()
    elif os.path.exists(args.model):
        # Load the model state
        model = torch.load(args.model, map_location=torch.device(args.device))
    else:
        # Start with a newly initialized model
        model = PathogenNet()
        print("Warning: Starting with a new model.")
        torch.save(model, args.model)
    return model

class Node():
    def __init__(self, state, parent, is_me, policy_prob=1.0):
        self.action = None
        self.state = state
        self.policy = None
        self.w = 0
        self.n = 0
        self.p = policy_prob
        self.parent = parent
        self.child_nodes = {}
        self.is_me = is_me
        self.temporary = False

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

    def puct(self, t):
        explore = self.p*np.sqrt(t)/(1 + self.n)
        if self.n == 0:
            return explore
        else:
            return (self.w / self.n) + explore

class NullModel:
    def __init__(self):
        pass

    def eval(self, *args):
        pass

    def train(self, *args):
        pass

    def __call__(self, input_tensor):
        # Assuming input_tensor shape is [N, 10]
        N = input_tensor.size(0)
        self.p = torch.zero(N, TOTAL_POS)
        self.valid = torch.zero(N, TOTAL_POS)
        self.v = torch.zero(N, 1)

        return self.p, self.valid, self.v

class NullDataset:
    def seek(self, *args):
        pass

    def write(self, *args):
        pass

    def close(self):
        pass

class CounterKey:
    def __init__(self, counter):
        self.counter = counter

    def __hash__(self):
        return hash(tuple(sorted(self.counter.items())))

    def __eq__(self, other):
        return isinstance(other, CounterKey) and self.counter == other.counter

    def __repr__(self):
        return f"CounterKey({self.counter})"

    def __str__(self):
        return f"CounterKey({self.counter})"

    def __iter__(self):
        return iter(self.counter.items())

class RLSimAgent(Agent):
    def __init__(self, args, s, n):
        super().__init__(args)
        self.action = 255
        self.s = s
        self.args = args

        ### MCTS stuff
        # Most of the time this is True.
        # Only when we finish a bunch of MCTS simulations,
        # we make a real play by revert this flag.
        self.simulation = False
        # when num_trials becomes 0, simulation goes from True to False;
        # when simulation goes from False to True, reset num_trials;
        # decrease this value each time a run of MCTS is finished
        self.num_trials = args.trial_unit
        # I want it to "feel" the end game condition first.
        # the average steps of a game is around 30.
        self.delay = args.delay_unit
        # We are not like Go, where it is clear and definite who plays every action.
        self.is_me = True
        # The nodes
        self.root = None
        self.current_node = None
        # The record of the game
        self.dataset = open(args.dataset+"_"+args.side+"_"+str(n)+".log", 'wb') if args.dataset != "/dev/null" else NullDataset()
        self.dataset_counter = 0

        # initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        if not torch.cuda.is_available():
            print('Warning: Use CPU')

        # We will only use this model for inference/simulation, at this phase
        self.model = init_model(args)
        self.model.eval()

        while self.play():
            continue
        if self.record is not None:
            self.record.close()

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
        if result == 0:
            state = np.frombuffer(self.current_node.state[4:], dtype=np.uint8)
            state = torch.from_numpy(np.copy(state)).float().unsqueeze(0).to(self.device)
            _, _, w = self.model(state)
        else:
            w = result
        w = float(w) * coef
        self.current_node.update(w)
        self.current_node = self.root
        return

    def candidates_to_prob(self):
        # sum up all the trials
        sum = 0
        # count each moves
        array = np.zeros(TOTAL_POS, dtype=np.float32)
        for i in self.current_node.child_nodes:
            # Since we have fast-forwarded most of the SetMarkers,
            # some nodes are marked with `temporary`.
            n = self.current_node.child_nodes[i].n
            if self.current_node.child_nodes[i].temporary:
                continue
            # Still count all of them
            if isinstance(i, CounterKey):
                for action, count in i:
                    sum = sum + count*n
                    index = action if action < BOARD_POS else action - MAP_POS_OFFSET
                    array[index] = array[index] + count*n
            else:
                index = i if i < BOARD_POS else i - MAP_POS_OFFSET
                array[index] = array[index] + n
                sum = sum + n

        return array/sum

    def analyze(self, data):
        if ord('E') == data[0]:
            print("This doesn't make any sense. Check it!")
            sys.exit(255)
        elif data[0:4] in (b'Ix0b'):
            # Let's fast forward all the SetMarkers in this tree search
            self.send_special(QUERY)
            self.action = random.choice(self.candidate)
            if self.delay > 0:
                return
            # Mark the node as temporary
            self.current_node.temporary = True

            # Form the corresponding index, as a CounterKey(Counter) object
            prev = self.current_node.action
            new_index = Counter()
            if isinstance(prev, Counter):
                new_index.update(prev)
            else:
                new_index.update([prev])
            new_index.update([self.action])

            # if exist, use it; if not, create it
            try:
                self.current_node = self.current_node.parent.child_nodes[CounterKey(new_index)]
            except KeyError:
                self.current_node.parent.child_nodes[CounterKey(new_index)] = Node(None, self.current_node.parent, None)
                self.current_node = self.current_node.parent.child_nodes[CounterKey(new_index)]
            finally:
                self.current_node.action = new_index
                assert not self.current_node == False, "Fail to get a corresponding node"

            return
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

            if self.dataset_counter > 0 and data[0:4] in (b'Ix04', b'Ix05'):
                i = 0
                WIN = np.ndarray([1], dtype=np.float32)
                WIN[0] = 1.0
                LOSE = np.ndarray([1], dtype=np.float32)
                LOSE[0] = -1.0
                while i < self.dataset_counter:
                    self.dataset.seek(i * DATASET_UNIT + 4*(S+2*TOTAL_POS), 0)
                    self.dataset.write(WIN.tobytes() if data[0:4] in (b'Ix04') else LOSE.tobytes())
                    i = i + 1
                self.dataset.seek(self.dataset_counter * DATASET_UNIT - 1, 0)
                self.dataset.write(b'\n')
                self.dataset.close()
            return
        elif data[0:4] in (b'Ix01', b'Ix03', b'Ix07', b'Ix08'):
            if data[0:4] in (b'Ix07'):
                self.is_me = False
            elif data[0:4] in (b'Ix03', b'Ix08'):
                self.is_me = True
            else:
                if self.delay <= 0:
                    assert self.current_node.parent != None, "An Ix01 node is supposed to have a parent"
                    self.is_me = self.current_node.parent.is_me
                else:
                    pass
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

                # if self.current_node.child_nodes and len(self.current_node.child_nodes) <= 1:
                    # no point to SAVE such a root
                    # self.action = QUERY
                # else:
                    # Ideally most of the time this is trying simulated moves, so if
                    # it is not in a simulation, just enable it.
                    # We can see this SAVE as a special variant of a RETURN to the
                    # root state because this one has no child currently.
                self.root.parent = None
                self.action = SAVE
                self.num_trials = self.args.trial_unit
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
                    self.update(0)
                else:
                    self.action = QUERY

            self.send_special(self.action)
            if self.simulation and not self.current_node.child_nodes:
                self.state = np.frombuffer(self.current_node.state[4:], dtype=np.uint8)
                self.state = torch.from_numpy(np.copy(self.state)).float().unsqueeze(0).to(self.device)
                policy, _, _ = self.model(self.state)
                probabilities = torch.nn.functional.softmax(policy, dim=1).squeeze(0)
                for action in self.candidate:
                    self.current_node.child_nodes[action] = Node(None, self.current_node, None, probabilities[action if action < BOARD_POS else action - MAP_POS_OFFSET])
                # why would we backtracking for this single option?
                # because it can help us update every node's inferenced weight
                # if len(self.candidate) > 1:
                self.num_trials = self.num_trials - 1
                self.action = RETURN
                self.send_special(self.action)
                self.update(0)
        elif data[0:4] in (b'Ix09', b'Ix0a'):
            self.num_trials = self.num_trials - 1
            if self.num_trials <= 0:
                self.action = CLEAR
                self.simulation = False
            else:
                self.action = RETURN
            # Otherwise, there will be no state for this final leaf node
            self.current_node.state = data
            # Otherwise, there will be no sign for this final leaf node
            self.current_node.is_me = True
            self.update(-1 if data[0:4] == b'Ix09' else 1)
            self.send_special(self.action)
        else:
            print("What is this?")
            panic()

        self.action = None
        if self.delay > 0:
            self.action = random.choice(self.candidate)
        elif not self.simulation:
            # Get ready for the dataset file offset
            self.dataset.seek(self.dataset_counter * DATASET_UNIT, 0)

            # Make next action
            self.state = np.frombuffer(self.current_node.state[4:], dtype=np.uint8)
            self.state = torch.from_numpy(np.copy(self.state)).float().unsqueeze(0).to(self.device)
            self.dataset.write(self.state.cpu().numpy().tobytes()) # section 1: state
            policy, valid, value = self.model(self.state)
            probabilities = spice(torch.nn.functional.softmax(policy, dim=1).squeeze(0), TEMPERATURE)

            ctp = self.candidates_to_prob()
            # Maybe we just shouldn't rely on this one? not sure... at least this is not what the book does.
            # probabilities2 = spice(torch.nn.functional.softmax(torch.from_numpy(np.copy(ctp)).float().unsqueeze(0).to(self.device), dim=1).squeeze(0), TEMPERATURE)

            # Then, record the probabilities and the valid head. Once the game is
            # over, we can add value back.
            self.dataset.write(ctp.tobytes()) # section 2: policy
            self.dataset.write(shsor(self.candidate).tobytes()) # section 3: valid
            # section 4, value, is not known until the end of the game
            self.dataset_counter = self.dataset_counter + 1

            for action_index in probabilities:
                index = int(action_index)
                if index in self.candidate:
                    if index >= BOARD_POS:
                        self.action = index + MAP_POS_OFFSET
                    else:
                        self.action = index

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
            self.current_node = self.root.child_nodes[self.action]
            self.root = self.current_node

        elif self.candidate == None:
            print("This doesn't make any sense. Check it!")
            output(data)
            sys.exit(255)
        elif len(self.candidate) == 1:
            # while other shortcuts have been removed, this one is preserved because it
            # has no other side effects and a time saver, compared to the else block below.
            self.action = self.candidate[0]
            self.current_node = self.current_node.child_nodes[self.action]
        else:
            # [RL]
            # Run the child node who has the maximum PUCT
            max_action_puct = float('-inf');
            assert len(self.current_node.child_nodes.keys()) == len(list(self.candidate)), "Why???"
            for candidate in self.candidate:
                puct = self.current_node.child_nodes[candidate].puct(self.current_node.n)
                if max_action_puct < puct:
                    max_action_puct = puct
                    self.action = candidate

            self.current_node = self.current_node.child_nodes[self.action]

        if self.delay <= 0:
            self.current_node.action = self.action

def spice(x, t):
    # We don't really need the gradients, do we?
    x_tensor = x.clone().detach().requires_grad_(False)
    # x_tensor = torch.tensor(x, dtype=torch.float32)
    x_tensor = x_tensor ** (1 / t)
    return torch.multinomial(x_tensor / x_tensor.sum(), SPICE if SPICE <= len(x_tensor) else len(x_tensor))

# Regarding this naming, it is an unintetional typo feeded into chatGPT,
# and it keeps it. It was meant to be `a short utility`.
def shsor(input_bytes):
    # Initialize a one-hot encoded array of length 61 with zeros
    onehot_array = np.zeros(TOTAL_POS, dtype=np.float32)

    # Iterate over each byte in the input
    for byte in input_bytes:
        # Check if the byte is within the first range [0, 36)
        if 0 <= byte < BOARD_POS:
            onehot_array[byte] = 1.0
        # Check if the byte is within the second range [100, 125)
        elif 0 <= byte - MAP_POS_OFFSET - BOARD_POS < MAP_POS:
            onehot_array[byte - MAP_POS_OFFSET] = 1.0

    return onehot_array

