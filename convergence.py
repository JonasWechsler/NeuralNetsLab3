import hopfield

if __name__ == "__main__":
    x1 = [-1, -1,  1, -1,  1, -1, -1,  1]
    x2 = [-1, -1, -1, -1, -1,  1, -1, -1]
    x3 = [-1,  1,  1, -1, -1,  1, -1,  1]
    x1d = [ 1, -1,  1, -1,  1, -1, -1,  1]
    x2d = [ 1,  1, -1, -1, -1,  1, -1, -1]
    x3d = [ 1,  1,  1, -1,  1,  1, -1,  1]
    X0 = [x1, x2, x3]
    W = hopfield.weights(X0)
    X = [x1d, x2d, x3d]
    D = [hopfield.recall_until_stable(W, x) for x in X]
    print(D)
    for x, exp in zip(X, X0):
        hopfield.test_expected(W, x, exp)
