import random
import os
import argparse

# Define constants
entry_size = 4096  # Each entry is 4096 bytes
total_entries = 50075  # Total number of entries in the dataset
entries_to_keep = 50000  # Number of entries to keep after shuffling

# Input and output file paths
parser = argparse.ArgumentParser(description='Dataset shuffler for the training of Pathogen agents')
parser.add_argument('-i', '--input', type=str, help='raw dataset path', default='/dev/null')
parser.add_argument('-o', '--output', type=str, help='output dataset path', default='/dev/null')
args = parser.parse_args()
input_file_path = args.input
output_file_path = args.output

# Step 1: Read the entire dataset into memory
with open(input_file_path, 'rb') as infile:
    data = infile.read()

# Step 2: Split the data into entries
entries = [data[i:i+entry_size] for i in range(0, len(data), entry_size)]

# Step 3: Randomly shuffle the entries
random.shuffle(entries)

# Step 4: Select the first 3000 entries
selected_entries = entries[:entries_to_keep]

# Step 5: Write the selected entries to a new file
with open(output_file_path, 'wb') as outfile:
    for entry in selected_entries:
        outfile.write(entry)

# Optional: Confirm the new file size
expected_size = entries_to_keep * entry_size
actual_size = os.path.getsize(output_file_path)

print(f'New file size: {actual_size} bytes')
print(f'Expected size: {expected_size} bytes')

if actual_size == expected_size:
    print("File was successfully created and is the expected size.")
else:
    print("There was an issue with the file creation.")

# The script will automatically close the files after the 'with' block exits

