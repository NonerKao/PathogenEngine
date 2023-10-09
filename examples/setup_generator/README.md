
# Quick Start

```
for _ in $(seq 1 100); do
    # Generate a UUID
    id=$(uuidgen)

    # Use the tool with the generated UUID as seed and save the output
    cargo run --example setup_generator --seed "$id" --save "output/$id.sgf"
done
```
