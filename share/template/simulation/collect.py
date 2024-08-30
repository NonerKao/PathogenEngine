import sys
import os
import re
import statistics

def get_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
        return size
    except OSError as e:
        print(f"Error: {e}")
        return None

def is_multiple_of_4096(size):
    return size % 4096 == 0

def collect_log_files(directory, op):
    # Regex pattern to extract the color and steps
    pattern2 = re.compile(r'\.log(\.\d+)?$')

    # Walk through the directory and process each matching file
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern2.search(file):
                file_path = os.path.join(root, file)
                file_size = get_file_size(file_path)
                if file_size is not None:
                    if not is_multiple_of_4096(file_size):
                        continue;
                with open(file_path, 'rb') as f:
                    op.write(f.read())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stat.py <path_to_records_directory> <output>")
        sys.exit(1)

    records_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        with open(output_path, 'wb') as op:
            collect_log_files(records_path, op)
    except IOError as e:
        print(f"Error: {e}")

    size = get_file_size(output_path)
    print(f"Collected {size / 4096:.2f} entries.")
