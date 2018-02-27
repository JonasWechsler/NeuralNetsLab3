import numpy as np

def weights(X):
    N = len(X[0])
    P = len(X)
    W_ij = lambda i, j: (1/N)*sum(X[m][i] * X[m][j] for m in range(P))
    W = [[W_ij(i, j) for i in range(N)] for j in range(N)]
    return np.array(W)

def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def recall(W, in_x):
    N = len(in_x)
    x_i = lambda i: sign(sum(W[i][j]*in_x[j] for j in range(N)))
    out_x = [x_i(i) for i in range(N)]
    return np.array(out_x)

if __name__ == "__main__":
    def test_stable(W, x):
        result = recall(W, x)
        if (result == x).all():
            print("Pass")
        else:
            print("Fail", x, result)

    x1 = [-1, -1,  1, -1,  1, -1, -1,  1]
    x2 = [-1, -1, -1, -1, -1,  1, -1, -1]
    x3 = [-1,  1,  1, -1, -1,  1, -1,  1]
    X = [x1, x2, x3]
    W = weights(X)
    for x in X:
        test_stable(W, x)
