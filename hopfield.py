import numpy as np

def weights(X):
    N = len(X[0])
    P = len(X)
    W_ij = lambda i, j: (1/N)*sum(X[m][i] * X[m][j] for m in range(P))
    W = [[W_ij(i, j) for i in range(N)] for j in range(N)]
    return np.array(W)

def sign(x):
    if x >= 0: return 1
    if x < 0: return -1
    return 0

def recall(W, in_x):
    N = len(in_x)
    x_i = lambda i: sign(sum(W[i][j]*in_x[j] for j in range(N)))
    out_x = [x_i(i) for i in range(N)]
    return np.array(out_x)


def recall_until_stable(W, x, max_iterations=1000):
    for _ in range(max_iterations):
        new_x = recall(W, x)
        if (new_x == x).all():
            return new_x
        x = new_x
    return -1

def test_stable(W, x):
    result = hopfield.recall(W, x)
    if (result == x).all():
        print("Pass")
    else:
        print("Fail", x, result)

