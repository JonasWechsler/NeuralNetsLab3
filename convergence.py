import hopfield

if __name__ == "__main__":
    x1 = [-1, -1,  1, -1,  1, -1, -1,  1]
    x2 = [-1, -1, -1, -1, -1,  1, -1, -1]
    x3 = [-1,  1,  1, -1, -1,  1, -1,  1]
    x1d = [ 1, -1,  1, -1,  1, -1, -1,  1]
    x2d = [ 1,  1, -1, -1, -1,  1, -1, -1]
    x3d = [ 1,  1,  1, -1,  1,  1, -1,  1]
    X = [x1, x2, x3]
    W = hopfield.weights(X)
    D = [hopfield.recall_until_stable(W, x) for x in X]
    print(D)
    for x in D:
        hopfield.test_stable(W, x)
