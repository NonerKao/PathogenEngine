import random
import os
import sys
import argparse

def shuf_binary_file(input_file_path, output_file_path, num_entries):
    # Define constants
    entry_size = 4096  # Each entry is 4096 bytes
    
    # Step 1: Read the entire dataset into memory
    with open(input_file_path, 'rb') as infile:
        data = infile.read()
    
    # Step 2: Split the data into entries
    entries = [data[i:i+entry_size] for i in range(0, len(data), entry_size)]
    
    # Step 3: Randomly shuffle the entries
    random.shuffle(entries)
    
    # Step 4: Select the first 3000 entries
    selected_entries = entries[:num_entries]
    
    # Step 5: Write the selected entries to a new file
    with open(output_file_path, 'wb') as outfile:
        for entry in selected_entries:
            outfile.write(entry)
    
    # Optional: Confirm the new file size
    expected_size = num_entries * entry_size
    actual_size = os.path.getsize(output_file_path)
    
    print(f'New file size: {actual_size} bytes')
    print(f'Expected size: {expected_size} bytes')
    
    if actual_size == expected_size:
        print("File was successfully created and is the expected size.")
    else:
        print("There was an issue with the file creation.")
    
    # The script will automatically close the files after the 'with' block exits

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python dataset_shuffler.py <file_name> <output_name> <num_entries>")
    else:
        file_name = sys.argv[1]
        output_name = sys.argv[2]
        num_entries = int(sys.argv[3])
        
        shuf_binary_file(file_name, output_name, num_entries)

