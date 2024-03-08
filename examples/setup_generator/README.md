
# Generate a bunch of cases

```
for _ in $(seq 1 100); do
    # Generate a UUID
    id=$(uuidgen)

    # Use the tool with the generated UUID as seed and save the output
    cargo run --example setup_generator -- --mode sgf --seed "$(echo $id | sed -e 's/-//g')" --save "output/$id.sgf"
done
```
