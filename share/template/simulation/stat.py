import sys
import os
import re
import statistics

def parse_sgf_files(directory):
    # Initialize counters and lists for steps
    total_lines = 0
    w_wins = 0
    b_steps = []
    w_steps = []

    # Regex pattern to extract the color and steps
    pattern = re.compile(r'.*\[(B|W)\+(\d+)\].*')
    pattern2 = re.compile(r'\.sgf(\.\d+)?$')

    # Walk through the directory and process each matching file
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern2.search(file):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        match = pattern.match(line)
                        if match:
                            total_lines += 1
                            color = match.group(1)
                            steps = int(match.group(2))
                            if color == 'W':
                                w_wins += 1
                                w_steps.append(steps)
                            elif color == 'B':
                                b_steps.append(steps)

    # Calculate W's winrate
    w_winrate = w_wins / total_lines if total_lines > 0 else 0

    # Calculate averages and standard deviations
    b_avg_steps = statistics.mean(b_steps) if b_steps else 0
    b_std_dev = statistics.stdev(b_steps) if len(b_steps) > 1 else 0

    w_avg_steps = statistics.mean(w_steps) if w_steps else 0
    w_std_dev = statistics.stdev(w_steps) if len(w_steps) > 1 else 0

    # Output the results
    print(f"Doctor's winrate: {w_winrate*100:.2f}%")
    print(f"Plague's steps to win: {b_avg_steps:.2f} +/- {b_std_dev:.2f}")
    print(f"Doctor's steps to win: {w_avg_steps:.2f} +/- {w_std_dev:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stat.py <path_to_records_directory>")
        sys.exit(1)

    records_path = sys.argv[1]
    parse_sgf_files(records_path)
