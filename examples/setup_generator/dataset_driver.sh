#!/bin/bash
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -n, --num-setup INT    Number of setups (default: 1)"
    echo "  -o, --output DIR       Directory to store artifacts (default: ./test)"
    echo "  -j, --jobs INT         Parallelism (default: 8)"
    echo "  -t, --trial INT        Number of games that the evaluator plays through (default: 10000)"
    echo "  -h, --help             Display this help and exit"
}

if [[ "$#" -eq 0 ]]; then
    print_help
    exit 0
fi

# Default values
SETUPS=1
OUTPUT="test"
TRIAL=10000
RAYON_NUM_THREADS=8

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num-setup) SETUPS="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        -j|--jobs) RAYON_NUM_THREADS="$2"; shift 2 ;;
        -t|--trial) TRIAL="$2"; shift 2 ;;
	-h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

mkdir -p $OUTPUT

# Example usage of the output file and directory

for ((i=1; i<=$SETUPS; i++)); do
    echo "Iteration $i of $SETUPS"
    seed0=$(uuidgen)
    echo $seed0 >> "$OUTPUT/list"
    cargo run --release --example setup_generator -- --mode sgf --seed "$(echo $seed0 | sed -e 's/-//g')" --save "$OUTPUT/$seed0.sgf" 2>/dev/null
    cargo run --release --example setup_generator -- --mode dataset --seed "$(echo $seed0 | sed -e 's/-//g')" --save "$OUTPUT/"$SETUPS"s_"$TRIAL"t.bin" 2>/dev/null
    
    # This is tricky. The byte output is verified by piping to
    #
    #     python -c "import sys; import struct; print(struct.unpack('f', sys.stdin.buffer.read(4))[0])"
    #
    # so there will be 4 bytes more appended as the label.
    cargo run --release --example setup_evaluator -- \
        --load $OUTPUT/$seed0.sgf --iter $TRIAL 2>/dev/null --seed "$(echo $seed0 | sed -e 's/-//g')" >> "$OUTPUT/"$SETUPS"s_"$TRIAL"t.bin"
done
