# Exercise 4.1 - Neuron dengan BIAS, bisa OR atau AND
from numpy import exp, array, random, dot, hstack, ones

def sigmoid(z):
    return 1 / (1 + exp(-z))

def train(target, seed=1, iterations=50000, lr=0.1):
    inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
    outputs = array([target]).T
    X = hstack([inputs, ones((4, 1))])          # kolom ke-3 = bias (selalu 1)
    random.seed(seed)
    weights = 2 * random.random((3, 1)) - 1
    for _ in range(iterations):
        out = sigmoid(dot(X, weights))
        weights += lr * dot(X.T, (outputs - out) * out * (1 - out))
    return weights

def neuron(x, weights):
    return sigmoid(dot(hstack([x, 1]), weights))

# Urutan data latih: [0,0], [1,1], [1,0], [0,1]
for name, target in [("OR", [0, 1, 1, 1]), ("AND", [0, 1, 0, 0])]:
    w = train(target)
    print(name, "weights:", w.ravel().round(3))
    for p in ([0, 0], [0, 1], [1, 0], [1, 1]):
        print("  ", p, "->", round(float(neuron(array(p), w)[0]), 4))
