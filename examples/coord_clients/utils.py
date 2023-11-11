def output(data):
    output_env(data)
    print()
    output_map(data[324:374])
    print()
    print(data[374:387])

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
