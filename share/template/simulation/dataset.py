import sys
import os
import struct
import random

def split_binary_file(file_name, train_count, train_output_name, eval_count, eval_output_name):
    # Define the size of each entry in bytes
    entry_size = 4096
    
    # Open the input binary file
    with open(file_name, 'rb') as f:
        # Calculate the total number of entries
        f.seek(0, 2)  # Move to the end of the file
        file_size = f.tell()
        if file_size % entry_size != 0:
            print(f"Error: The file size ({file_size} bytes) is not divisible by {entry_size}.")
            print("The file may be corrupted or not in the expected format.")
            return
        total_entries = file_size // entry_size

        # Validate the train_count
        if train_count + eval_count > total_entries:
            print(f"Error: The number of training entries ({train_count}) exceeds the total number of entries ({total_entries}).")
            return

        # Rewind to the start of the file
        f.seek(0)
        
        # Read the training set
        with open(train_output_name, 'wb') as train_f:
            for _ in range(train_count):
                train_f.write(f.read(entry_size))
        
        # Read the evaluation set
        with open(eval_output_name, 'wb') as eval_f:
            for _ in range(eval_count):
                eval_f.write(f.read(entry_size))
        
    print(f"Total entries: {total_entries}")
    print(f"Training set written to '{train_output_name}' with {train_count} entries.")
    print(f"Evaluation set written to '{eval_output_name}' with {eval_count} entries.")

def shuf_binary_file(input_file_path, num_entries, output_file_path):
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
    if len(sys.argv) != 6:
        print("Usage: python dataset.py <file_name> <train_count> <train_output_name> <eval_count> <eval_output_name>")
    else:
        file_name = sys.argv[1]
        train_count = int(sys.argv[2])
        train_output_name = sys.argv[3]
        eval_count = int(sys.argv[4])
        eval_output_name = sys.argv[5]
        
        temp = '/tmp/tmp.dataset'
        split_binary_file(file_name, train_count, temp, eval_count, eval_output_name)
        shuf_binary_file(temp, train_count, train_output_name)
