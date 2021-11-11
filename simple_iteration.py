import numpy as np


def read_system():
    inp = open("input2.txt", "r")
    matr = inp.readlines()
    A = []
    B = []
    for i in range(len(matr)):
        row = matr[i].split()
        A.append(list(map(float, row[:-1])))
        B.append(float(row[-1]))
    A = np.array(A)
    B = np.array(B)
    B = B.reshape(len(B), 1)

    return A, B


def simple_iteration(A, B, eps=1e-5):
    nA = np.copy(A)
    nB = np.copy(B)
    n = len(A)
    for i in range(n):
        nA[i] = A[i] / (-1.0 * A[i, i])
        nB[i] = B[i, 0] /  A[i, i]
        nA[i, i] = 0

    x_1 = np.copy(nB)
    x_2 = nA @ x_1 + nB
    print(nA)
    while np.linalg.norm(B - A @ x_2) > eps:
        print(x_2)
        x_1 = x_2
        x_2 = nA @ x_1 + nB

    return x_2


if __name__ == "__main__":
    A, B = read_system()
    x = simple_iteration(A, B)
    print(A @ x)
