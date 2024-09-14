import os
import sys
import random

def split_file(file_path, train_ratio=0.875, align_index=60):
    # Get the base name without extension
    base_name = os.path.splitext(file_path)[0]
    entry_size = 4096  # Each entry is 4096 bytes
    
    # Step 1: Read the entire dataset into memory
    with open(file_path, 'rb') as infile:
        data = infile.read()
    
    # Step 2: Split the data into entries
    entries = [data[i:i+entry_size] for i in range(0, len(data), entry_size)]
    
    # Step 3: Randomly shuffle the entries
    random.shuffle(entries)

    # Determine the split index
    total_index = len(entries) // align_index * align_index
    split_index = int(len(entries) * train_ratio) // align_index * align_index

    # Create train and test file paths
    train_file_path = f"{base_name}.train.bin"
    test_file_path = f"{base_name}.eval.bin"

    # Step 4: Select the first 3000 entries
    selected_entries = entries[:split_index]
    
    # Step 5: Write the selected entries to a new file
    with open(train_file_path, 'wb') as f_train:
        for entry in entries[:split_index]:
            f_train.write(entry)

    with open(test_file_path, 'wb') as f_test:
        for entry in entries[split_index:total_index]:
            f_test.write(entry)
    
    print(f"Created {train_file_path} and {test_file_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python new_dataset.py <file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        sys.exit(1)

    split_file(file_path)
