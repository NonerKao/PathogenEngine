import socket

class II_Agent:
    def __init__(self, fraction):
        self.fraction = fraction
        if fraction == "White":
            self.port = 6241
            self.action = 40
        elif fraction == "Black":
            self.port = 3698
            self.action = 80
        else:
            raise ValueError("Unknown fraction!")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', self.port))

    def play(self):
        data = self.s.recv(154)
        game_state = data[:150]
        status_code = data[150:]
        # Replace with logic to determine the best action

        action = self.get_best_action()
        self.s.sendall(bytes([action]))

        # Use game_state and status_code for further actions
        # ...

    def get_best_action(self):
        # Here, we should use the neural network to select an action.
        # Just returning a dummy action for now.
        self.action = self.action-1;
        return self.action

# Create agent instances
white_agent = II_Agent("White")
black_agent = II_Agent("Black")

while True:
    # For demonstration, starting white agent to play
    white_agent.play()
    black_agent.play()

