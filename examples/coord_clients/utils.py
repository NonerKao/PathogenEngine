from constant import *

def output(data):
    print(data[0:CODE_DATA])
    print()
    output_env(data[CODE_DATA:CODE_DATA+BOARD_DATA])
    print()
    output_map(data[CODE_DATA+BOARD_DATA:CODE_DATA+BOARD_DATA+MAP_DATA])

def output_env(data):
    for i in range(6):
        for j in range(6):
            print(f"({i}, {j})", end=' ')
            print(data[i*6*9 + j*9 : i*6*9 + j*9 + 9])

def output_map(data):
    # The second loop outputs the 5*5*2 bytes
    for i in range(5):
        for j in range(5):
            print(f"({i-2}, {j-2})", end=' ')
            print(data[i*5*2 + j*2 : i*5*2 + j*2 + 2])
