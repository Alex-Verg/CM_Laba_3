import numpy as np


def read_system():
    inp = open("input.txt", "r")
    matr = inp.readlines()
    A = []
    B = []
    for i in range(len(matr)):
        A.append(list(map(float, matr[i].split()[:-1])))
        B.append(float(matr[i].split()[-1]))
    A = np.array(A)
    B = np.array(B)
    B = B.reshape(len(B), 1)

    return A, B


if __name__ == "__main__":
    A, B = read_system()
