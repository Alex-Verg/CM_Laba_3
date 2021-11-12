import numpy as np
import time


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
    while np.linalg.norm(B - A @ x_2) > eps:
        #print(x_2)
        x_1 = x_2
        x_2 = nA @ x_1 + nB

    return x_2


def gradient_boosting(A, B, eps=1e-5):
    x_2 = np.copy(B)
    x_1 = np.zeros_like(B)
    r = np.zeros_like(B)
    n = len(B)
    while np.linalg.norm(B - A @ x_2) > eps:
        x_1 = x_2
        for i in range(n):
            sum1 = 0
            sum2 = 0
            for j in range(n):
                for k in range(n):
                    sum1 += A[i, j] * A[j, k] * x_1[k]
                sum2 += A[i, j] * B[j, 0]
            r[i, 0] = - sum1 + sum2
        alpha = (np.sum(r ** 2)) / np.sum((A @ r) ** 2)
        x_2 = x_1 + alpha * r

    return x_2


if __name__ == "__main__":
    A, B = read_system()
    st = time.time()
    x = simple_iteration(A, B)
    fin = time.time()
    print(x, fin-st)
    st = time.time()
    x2 = gradient_boosting(A, B)
    fin = time.time()
    print(x2, fin-st)
