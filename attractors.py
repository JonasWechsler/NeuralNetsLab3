import hopfield
import itertools
import numpy as np

if __name__ == "__main__":
    N = 8
    x1 = [-1, -1,  1, -1,  1, -1, -1,  1]
    x2 = [-1, -1, -1, -1, -1,  1, -1, -1]
    x3 = [-1,  1,  1, -1, -1,  1, -1,  1]
    X = [x1, x2, x3]
    W = hopfield.weights(X)
    attractors = {}
    for seq in itertools.product([-1, 1], repeat=N):
        stable = hopfield.recall_until_stable(W, np.array(seq))
        stable.flags.writeable = False
        stringified = str(stable)
        if stringified not in attractors:
            attractors[stringified] = 1
        else:
            attractors[stringified] += 1
    print("\n".join(["{}:{}".format(key, attractors[key]) for key in attractors]))
    print("There are {} attractors".format(len(attractors)))
