
## Quick Start

```
seq 2601 2700 | parallel -j8 RUST_BACKTRACE=1 cargo run --release --example setup_evaluator -- -l setups/34fe560c-4a35-428e-b2ee-5bc15bf88ba9.sgf --seed {} -i 10000 2>/dev/null
```
