
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
