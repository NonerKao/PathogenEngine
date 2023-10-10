
# Architecture

# Usage

## Build

```
$ cargo build
```

## Test

```
$ cargo test core -- --nocapture
```

# Specs

## Status code

The format of the status code should be a 4-letter short string. The first letter should be `E`, `W`, or `I`, indicating error, warning, or info respectively. The second must be `x`. The final two should be two hexadecimal digit (regex: [0-9a-f]). The encoding is in `status_code.rs`.

## `Game` as a state machine

## `Action` as a state machine

* **SetMap** x 1
   * On success, determine if it goes to **Lockdown**
   * On failure, restart **SetMap**. Note that, it is immediately decidable whether a move is possible with a chosen new map position.
* (Optional) **Lockdown** x 1
   * On success, go to **SetCharacter**
   * On failure, restart **Lockdown**
* **SetCharacter** x 1
   * On success, go to **BoardMove**
   * On failure, restart **SetCharacter**
* **BoardMove** x 5 (at most)
   * On success, go to the next **BoardMove**
   * On done, go to **SetMarkers**
   * On failure, restart **BoardMove**
* **SetMarkers** x 5 (at most)
   * On success, go to the next **SetMarkers**
   * On done, go to **Done**
   * On failure, restart **SetMarkers**
* **Done**

