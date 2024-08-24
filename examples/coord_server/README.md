
# Quick Start

## Server

```
$ cargo run --example coord_server --load-dir <setup dir> [--batch <num>] [--save-dir <record dir>] 
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

Totally speaking, **809** bytes.

### The board (or `env` in the codes)

**288 = 6** (X-axis, from `[ax]` to `[fx]`) **x 6** (Y-axis) **x 8** (underworld, humanity, doctor character, plague character, doctor colony, plague colony, doctor marker, plague marker). Other than the markers, this tensor is mostly one-shot. The last extra marker is for set-marker session.

### The map

**50 = 5** (X-axis, from `[gx]` to `[kx]`) **x 5** (Y-axis) **x 2** (doctor marker, plague marker).

### Turn information

**25 = 5 x 5 x 1** (duplicate the marker of the camp).

### Control

The agent should know which sub-step it is currently in~~, so here is a one more one-hot array for tracking the status. The tricky part is the move-character and set-marker sessions that are of various length~~. Considering the perception ability of DNN, this will be re-formated as tensors.

**446 = 5 x 5 x 2** (set-map, lockdown) and **6 x 6 x 11** (set-character x 1, board-move x 5, and set-marker x 5). Note that in the board-move phase, the phase can end under 5 sub-moves. We fill the rest with the final sub-move (the destination). Hopefully, the DNN will be able to tell that the first pure zero layer represents the sub-move it should predict.

> Note: 0.6 introduces a huge change. We remove the whole content in Update with partial action previously.
