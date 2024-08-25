import sys
import struct

def split_binary_file(file_name, train_count, train_output_name, eval_output_name):
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
        if train_count > total_entries:
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
            for _ in range(total_entries - train_count):
                eval_f.write(f.read(entry_size))
        
    print(f"Total entries: {total_entries}")
    print(f"Training set written to '{train_output_name}' with {train_count} entries.")
    print(f"Evaluation set written to '{eval_output_name}' with {total_entries - train_count} entries.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python split_binary_file.py <file_name> <train_count> <train_output_name> <eval_output_name>")
    else:
        file_name = sys.argv[1]
        train_count = int(sys.argv[2])
        train_output_name = sys.argv[3]
        eval_output_name = sys.argv[4]
        
        split_binary_file(file_name, train_count, train_output_name, eval_output_name)

