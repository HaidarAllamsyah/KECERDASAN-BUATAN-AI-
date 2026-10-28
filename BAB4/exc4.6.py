# Exercise 4.6 - tambah satu blok Conv2D + MaxPooling + Dropout sebelum Flatten
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

def asli():
    m = Sequential([Input(shape=(28, 28, 1)),
        Conv2D(32, (5, 5), activation='relu'), MaxPooling2D(pool_size=(2, 2)), Dropout(0.2),
        Flatten(), Dense(128, activation='relu'), Dense(2, activation='softmax')])
    return m

def baru():
    m = Sequential([Input(shape=(28, 28, 1)),
        Conv2D(32, (5, 5), activation='relu'), MaxPooling2D(pool_size=(2, 2)), Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(pool_size=(2, 2)), Dropout(0.2),   # blok baru
        Flatten(), Dense(128, activation='relu'), Dense(2, activation='softmax')])
    return m

for nama, fn in [("Asli (Example 4.7)", asli), ("Dengan blok tambahan", baru)]:
    m = fn()
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(nama, "->", f"{m.count_params():,} parameter")
baru().summary()
