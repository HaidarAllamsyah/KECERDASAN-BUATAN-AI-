# Exercise 4.7 + 4.8 - bandingkan varian CNN pada MNIST
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

def build(filters=32, kernel=(5, 5), extra_block=False):
    m = Sequential([Input(shape=(28, 28, 1)),
                    Conv2D(filters, kernel, activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)), Dropout(0.2)])
    if extra_block:                                   # Exercise 4.8
        m.add(Conv2D(64, (3, 3), activation='relu'))
        m.add(MaxPooling2D(pool_size=(2, 2)))
        m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m

variants = {
    "A. Asli: 32 filter, kernel (5,5)":        dict(),
    "B. 64 filter, kernel (5,5)   [Ex 4.7]":   dict(filters=64),
    "C. 32 filter, kernel (3,3)   [Ex 4.7]":   dict(kernel=(3, 3)),
    "D. + blok Conv(64)+Pool+Drop [Ex 4.8]":   dict(extra_block=True),
}

print(f"{'Varian':42s}{'Parameter':>11s}{'Waktu':>9s}{'Error':>8s}")
for name, kw in variants.items():
    m = build(**kw)
    t0 = time.time()
    m.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=10, batch_size=200, verbose=0)
    waktu = time.time() - t0
    acc = m.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{name:42s}{m.count_params():>11,}{waktu:>8.0f}s{100 - acc * 100:>7.2f}%")
