
[Pathogen](https://boardgamegeek.com/boardgame/369862/pathogen) is an innovative two-player abstract game designed by [Kolor Deep Studio](https://boardgamegeek.com/boardgamepublisher/52369/kolor-deep-studio). This project is a fan-made digital implementation intended solely for personal use and not for commercial purposes. All rights remain with the original creators.

This is the legacy branch that will not be further maintained. **PRs against this branch will be dropped**.

## Quick Start

```
$ cargo build
```

An example use scenario: Plague plays in terminal A and Docker (the other player) connects from another terminal B. Terminal A hosts a game server with termion-based UI by 

```
$ ./target/debug/pathogen --ip 127.0.0.1 --port 8080 --doctor hjkl-bot
```

In Terminal B, the user doesn't run a specialized client for this game. Instead, they saves the terminal settings and then executes pure TCP clients like `netcat`. After the game ends, the saved tty information can be used to restore the terminal.

```
$ stty_settings=$(stty -g)
$ stty raw -echo
nc 127.0.0.1 8080
... (after the game ends)
$ stty $stty_settings
```

In both terminals, the user can use vim-like H-J-K-L for cursor and token movement, while some helper utility relies on the use of `TAB` and `SAPCE` keys.

## Features and Limitations

### SGF binding

The program comes with a Pathogen-specific SGF parser, and the gaming transactions also largely rely on the SGF syntax. A game can start from scratch or from a certain point recorded in a valid SGF file. To do the latter, one can use

```
$ ./target/debug/pathogen --load game_just_started.sgf
```

The program is not able to save game record.

### Game flow

The termion-based UI is manually crafted and there is some bugs.

The ending condition is not implemented.


