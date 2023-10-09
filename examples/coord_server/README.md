
# Quick Start

## Server

```
$ cargo run --example coord_server
```

## Client (The random client as an example)

```
$ python random.py -s Doctor
```

# Hacking

## Basic Flow

The `coord_*` server/client communicate with each other by a simple raw TCP. The flow is as follows,

1. Hello
   1. [Server] sends the **encoding of the game (E below)** and **hello status code (S below)** to both player.
   1. [Doctor] recieves E and S. Does nothing.
   1. [Plague] recieves E and S. Based on its own decision making process, sends a byte back, as **the action (A below)** it chooses.
1. Action
   1. [Server] processes it with `Action::add_*` methods. E should be crafted in a way that does NOT affect the real Game status, but reflects the update of the partial action is performed.  If the returning S is **some error**, the Action is wiped out and the player restarts from the beginning of the action. If the S is **OK**, then repeat this step until the action object is commitable. Finally, the S should be **DONE** for the action, or **WIN** if the player wins.
   1. [Server] commits the action.
   1. [Plague] recieves E and DONE, do nothing.
   1. [Doctor] recieves E and **TURN**, repeat Action but switching with Plague.
1. End
   1. [Server] sends E that represents the final results to both player, but S should be **WIN** and **LOSE** for each respectively.

Check `src/core/status_code.rs` for the details of status codes.

## Encoding of the game

Totally speaking, **387** bytes.

### The board (or `env` in the codes)

**324 = 6** (X-axis, from `[ax]` to `[fx]`) **x 6** (Y-axis) **x 9** (underworld, humanity, doctor character, plague character, doctor colony, plague colony, doctor marker, plague marker, extra marker). Other than the markers, this tensor is mostly one-shot. The last extra marker is for set-marker session.

> Future experiments: integrating underworld/humanity, doctor as positive and plague as opposite negative, etc.

### The map

**50 = 5** (X-axis, from `[gx]` to `[kx]`) **x 5** (Y-axis) **x 2** (doctor marker, plague marker).

### Control

The agent should know which sub-step it is currently in, so here is a one more one-hot array for tracking the status. The tricky part is the move-character and set-marker sessions that are of various length. 

We need **13 = 2** (map position and potentially Plague's position after lockdown) **+ 6** (character position and up to 5 steps) **+ 5** (markers). A few examples below:

* (1, 0, ...): this round, you (the agent) have to give me (the server) the map position for this move.
* (0, 1, 0, ...): this round, you have to give me the Plague's position after lockdown.
* (0, 0, 1, 1, 1, 1, 1, 1, 0, ...): total steps = 5, give me which character you want to move.
* (0, 0, 1, 1, 1, 1, 0, 0, 0, ...): total steps = 3. Give me your next step.
* (0, 0, 0, 0, 0, 1, 0, 0, 0, ...): total steps = 3, and only final step left.
* (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0): For Plague, set the markers.
* (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1): For Doctor, set the markers.

### Update with partial action

In the set-map session, just show as if they moved accordingly.

In the move-character session, set the character bit for the trajectory it goes through, excluding the end point.

In the set-marker session, use the last extra marker in the board to indicate the temporal setting.
