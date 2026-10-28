# Exercise 4.2 - MLP dengan 3 input (OR 3 input)
from itertools import product
from sklearn.neural_network import MLPClassifier

X = [list(map(float, p)) for p in product([0, 1], repeat=3)]       # 8 kombinasi
y = [1 if sum(p) > 0 else 0 for p in product([0, 1], repeat=3)]    # OR 3 input

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print("Data X:", X)
print("Label y:", y)
print("Prediksi data latih:", clf.predict(X))
print(clf.predict([[2., 2., 2.], [-1., -2., -1.]]))
print([coef.shape for coef in clf.coefs_])
