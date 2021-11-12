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


def print_matr(matr, exp=False):
    n, m = np.shape(matr)
    for i in range(n):
        for j in range(m):
            if exp:
                print(format(matr[i, j], " 7.6e"), end=" ")
            else:
                print(format(matr[i, j], " 7.6f"), end=" ")
        print()


def log_output(it, n, x, e, norm):
    for i in range(it):
        if it > 5 and 2 < i < it - 3:
            if i == 4:
                print(' . . . \n')
            continue
        else:
            print(i+1, 'iteration: ')
            print(' x  =', end=' ')
            print_matr(x[i].reshape(1, n))
            print(' E  =', end=' ')
            print_matr(e[i].reshape(1, n))
            print('|E| =', format(norm[i], " 7.6f"), end='\n\n')


def simple_iteration(A, B, eps=1e-5):
    nA = np.copy(A)
    nB = np.copy(B)
    n = len(A)
    for i in range(n):
        nA[i] = A[i] / (-1.0 * A[i, i])
        nB[i] = B[i, 0] / A[i, i]
        nA[i, i] = 0

    x_1 = np.copy(nB)
    x_2 = nA @ x_1 + nB
    vec = B - A @ x_2
    vec_n = np.linalg.norm(vec)
    x = [x_2]
    e = [vec]
    norm = [vec_n]
    it = 1
    while vec_n > eps:
        x_1 = x_2                   #recalculate x_2
        x_2 = nA @ x_1 + nB

        x.append(x_2)               #log x_2, B-A@x_2, |x_2|
        vec = B - A @ x_2
        vec_n = np.linalg.norm(vec)
        e.append(vec)
        norm.append(vec_n)
        it += 1

    log_output(it, n, x, e, norm)

    return x_2


def gradient_boosting(A, B, eps=1e-5):
    x_2 = np.copy(B)
    x_1 = np.zeros_like(B)
    r = np.zeros_like(B)
    n = len(B)
    x = []
    e = []
    norm = []
    it = 0
    while (vec_n := np.linalg.norm(vec := B - A @ x_2)) > eps:
        if it:
            x.append(x_2)
            e.append(vec)
            norm.append(vec_n)
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

        it += 1
    else:
        x.append(x_2)
        e.append(vec)
        norm.append(vec_n)

    log_output(it, n, x, e, norm)

    return x_2


if __name__ == "__main__":
    A, B = read_system()
    print("|" + "-" * 20 + "Simple iteration" + "-" * 20 + "|")
    x = simple_iteration(A, B)
    print('Result: x  =', end='')
    print_matr(x.reshape(1, len(x)))
    print(' '*8 + 'E  =', end='')
    print_matr((E := (B - A @ x).reshape(1, len(x))))
    print(' '*7 + '|E| =', format(np.linalg.norm(E), " 7.6f"), end='\n\n\n')

    print("|" + "-" * 20 + "Gradient boosting" + "-" * 19 + "|")
    x = gradient_boosting(A, B)
    print('Result: x  =', end='')
    print_matr(x.reshape(1, len(x)))
    print(' ' * 8 + 'E  =', end='')
    print_matr((E := (B - A @ x).reshape(1, len(x))))
    print(' ' * 7 + '|E| =', format(np.linalg.norm(E), " 7.6f"))
