import argparse
import struct

def byte_to_float(byte, divisor):
    if 65 <= byte <= 90:  # Capitalized alphabet
        return 1.0
    elif byte == 32 or byte == 0:  # Blank or \x00
        return 0.0
    else:
        return byte / divisor

def translate_file(input_file_path, output_file_path, divisor):
    with open(input_file_path, 'rb') as input_file, open(output_file_path, 'wb') as output_file:
        while byte := input_file.read(1):
            value = byte_to_float(ord(byte), divisor)
            output_file.write(struct.pack('f', value))

def main():
    parser = argparse.ArgumentParser(description="Binary-file-to-binary-file translator.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input binary file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output binary file.')
    parser.add_argument('--divisor', type=float, required=True, help='Divisor for non-special byte values.')

    args = parser.parse_args()

    translate_file(args.input, args.output, args.divisor)

if __name__ == '__main__':
    main()

