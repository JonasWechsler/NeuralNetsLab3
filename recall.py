import hopfield

if __name__ == "__main__":
    x1 = [-1, -1,  1, -1,  1, -1, -1,  1]
    x2 = [-1, -1, -1, -1, -1,  1, -1, -1]
    x3 = [-1,  1,  1, -1, -1,  1, -1,  1]
    X = [x1, x2, x3]
    W = hopfield.weights(X)
    for x in X:
        hopfield.test_stable(W, x)
