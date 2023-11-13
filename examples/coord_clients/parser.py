import difflib
import sys

def bytearray_diff(array1, array2):
    # Convert byte arrays to strings of hex representation
    str1 = ' '.join(format(x, '02x') for x in array1)
    str2 = ' '.join(format(x, '02x') for x in array2)

    # Use difflib to get a diff
    diff = list(difflib.ndiff(str1.split(), str2.split()))

    # Print or process the diff
    for line in diff:
        print(line)

def show_s(fa, s):
    lines = 13
    # channels
    for i in range(0, 9):
        # Y
        for j in range(0, 6):
            # X
            for k in range(0, 6):
                fa[i].write(str(s[k*6*9 + j*9 + i])+'   ')
            fa[i].write('\n\n')
    # padding
    for i in range(0, 9):
        for j in range(0, 22-lines):
            fa[i].write('\n')

if __name__ == "__main__":
    STATUS_SIZE = 374
    FLOW_SIZE = 13
    CODE_SIZE = 4
    count = 0
    fa = []
    for i in range(0, 9):
        fa.append(open('Q'+str(i), 'w'))
    with open('test.bin', 'rb') as file:
        while True:
            c_prev = file.read(CODE_SIZE)
            s_prev = file.read(STATUS_SIZE)
            f_prev = file.read(FLOW_SIZE)
            show_s(fa, s_prev)
            try:
                assert c_prev in (b'Ix03', b'Ix04', b'Ix05', b'Ix06')
            except AssertionError as error:
                print(c_prev)
                sys.exit(0)
            if c_prev in (b'Ix04', b'Ix05', b'Ix06'):
                count = 0
                continue
            while True:
                a = file.read(1)
                c_now = file.read(CODE_SIZE)
                s_now = file.read(STATUS_SIZE)
                f_now = file.read(FLOW_SIZE)
                if a == 255 or c_now in (b'', b'Ix01', b'Ix03'):
                    if c_now == b'':
                        break
                    print(c_now)
                    show_s(fa, s_now)
                    c_prev = c_now
                    s_prev = s_now
                    f_prev = f_now
                elif c_now in (b'Ix00',  b'Ix02', b'Ix04',  b'Ix05', b'Ix06'):
                    count = count + 1
                    print(count)
                    show_s(fa, s_now)
                elif c_now in (b'Ix04',  b'Ix05', b'Ix06'):
                    a = file.read(1)
                    assert a == 255
                    break
                else:
                   try:
                       assert s_now == s_prev
                   except AssertionError as error:
                       print(c_now)
                       bytearray_diff(s_prev, s_now)
                       sys.exit(0)

    for i in range(0, 9):
        fa[i].close()
