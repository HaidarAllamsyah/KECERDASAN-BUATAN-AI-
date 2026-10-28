# Exercise 4.3 - bandingkan Dense 10 neuron vs 5 neuron (waktu training & akurasi)
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

whole_data = load_breast_cancer()
X_tr, X_te, y_train, y_test = train_test_split(
    whole_data.data, whole_data.target, test_size=0.3, random_state=7)

scaler = StandardScaler().fit(X_tr)          # normalisasi (fit hanya dari data latih)
datasets = {"TANPA normalisasi": (X_tr, X_te),
            "DENGAN normalisasi": (scaler.transform(X_tr), scaler.transform(X_te))}

def run(units, X_train, X_test, seed):
    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    t0 = time.time()
    model.fit(X_train, y_train, batch_size=50, validation_split=0.2, epochs=100, verbose=0)
    waktu = time.time() - t0
    mse = model.evaluate(X_test, y_test, verbose=0)[0]
    pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    return model.count_params(), waktu, mse, (pred == y_test).mean()

for nama, (Xa, Xb) in datasets.items():
    print("==", nama)
    for units in (10, 5):
        hasil = [run(units, Xa, Xb, s) for s in (1, 2, 3)]
        p = hasil[0][0]
        print(f" units={units:2d} params={p} waktu_rata2={np.mean([h[1] for h in hasil]):.1f}s "
              f"mse_rata2={np.mean([h[2] for h in hasil]):.4f} "
              f"akurasi_per_seed={[round(float(h[3]),3) for h in hasil]}")
