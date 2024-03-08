#!/bin/bash
pad_spectrum() {
    local min="$1"
    local max="$2"

    # Read input and update the counts array
    i=$min
    while read -d ' ' count; do
        read index
        while [ $index != $i ]; do
            echo 0
            i=$(($i+1))
        done
        echo $count
        i=$(($i+1))
        if [ $i -ge $max ]; then
            break
        fi
    done
    while [ $i -le $max ]; do
        echo 0
        i=$(($i+1))
    done
}

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -i, --iterations INT   Number of iterations for internal logic (default: 1)"
    echo "  -o, --output DIR       Directory to store artifacts (default: current directory)"
    echo "  -b, --batch INT        Random trial size of the evaluator (default: 1)"
    echo "  -h, --help             Display this help and exit"
}

if [[ "$#" -eq 0 ]]; then
    print_help
    exit 0
fi

# Default values
ITERATIONS=1
OUTPUT="test"
BATCH=1
MASS=3890

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--iterations) ITERATIONS="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        -b|--batch) BATCH="$2"; shift 2 ;;
	-h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

mkdir -p $OUTPUT

# Example usage of the output file and directory

for ((i=1; i<=$ITERATIONS; i++)); do
    echo "Iteration $i of $ITERATIONS"
    seed0=$(uuidgen)
    echo $seed0 >> "$OUTPUT/list"
    cargo run --release --example setup_generator -- --mode sgf --seed "$(echo $seed0 | sed -e 's/-//g')" --save "$OUTPUT/$seed0.sgf" 2>/dev/null
    cargo run --release --example setup_generator -- --mode dataset --seed "$(echo $seed0 | sed -e 's/-//g')" --save "$OUTPUT/data.bin" 2>/dev/null
    
    for ((j=1; j<=$BATCH; j++)); do
        echo $(uuidgen | tr -d -)
    done | parallel -j8 \
        "cargo run --release --example setup_evaluator -- \
        --load $OUTPUT/$seed0.sgf --iter $MASS --seed {}" 2>/dev/null | sort -n | uniq -c | sort -n -k2 | sed -e 's/^[[:space:]]*//' | \
        pad_spectrum 0 $MASS | xargs printf "\\\\\\\\x""%x" | xargs echo -n -e >> "$OUTPUT/data.bin"

done

python dataset_translate.py --input $OUTPUT/data.bin --output $OUTPUT/"$ITERATIONS"s_"$BATCH"b.bin --divisor $BATCH
