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

def recall_sequentially(W, x, max_iterations=5000, snapshot=500):
    in_x = list(x)
    N = len(in_x)
    x_i = lambda i: sign(sum(W[i][j]*in_x[j] for j in range(N)))
    out_array = []
    for _ in range(max_iterations):
        choice = np.random.randint(0, len(in_x))
        in_x[choice] = x_i(choice)
        if not _ % snapshot:
            out_array.append(list(in_x))
    return out_array



def recall_until_stable(W, x, max_iterations=1000):
    for _ in range(max_iterations):
        new_x = recall(W, x)
        if (new_x == x).all():
            return new_x
        x = new_x
    return -1

def test_stable(W, x):
    result = recall(W, x)
    if (result == x).all():
        print("Pass")
    else:
        print("Fail", x, result)

def test_expected(W, x, exp):
    result = recall_until_stable(W, x)
    if (result == exp).all():
        print("Pass")
    else:
        print("Fail", result, exp)

def energy(W, x):
    N = len(W)
    E_i = lambda i: sum(W[i][j]*x[j]*x[i] for j in range(N))
    return -1*sum(E_i(i) for i in range(N))



