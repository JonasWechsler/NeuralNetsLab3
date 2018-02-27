import hopfield
import itertools
import numpy as np

def get_weights():
    x1 = [-1, -1,  1, -1,  1, -1, -1,  1]
    x2 = [-1, -1, -1, -1, -1,  1, -1, -1]
    x3 = [-1,  1,  1, -1, -1,  1, -1,  1]
    X = [x1, x2, x3]
    return hopfield.weights(X)

def get_attractors():
    N = 8
    W = get_weights()
    attractors = {}
    attractors_set = set()
    for seq in itertools.product([-1, 1], repeat=N):
        stable = hopfield.recall_until_stable(W, np.array(seq))
        stable.flags.writeable = False
        stringified = str(stable)
        attractors_set.add(tuple(stable))
        if stringified not in attractors:
            attractors[stringified] = 1
        else:
            attractors[stringified] += 1
    return attractors_set

if __name__ == "__main__":
    attractors = get_attractors()
    print("\n".join(["{}:{}".format(key, attractors[key]) for key in attractors]))
    print("There are {} attractors".format(len(attractors)))
