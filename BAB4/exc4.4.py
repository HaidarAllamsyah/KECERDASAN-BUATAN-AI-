# Exercise 4.4 - LeNet dengan jumlah filter berbeda
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Input

def build_lenet(f1=6, f2=16):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(filters=f1, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=f2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model

for f1, f2 in [(3, 8), (6, 16), (12, 32), (32, 64)]:
    print(f"filter ({f1:2d},{f2:2d}) -> total parameter = {build_lenet(f1, f2).count_params():,}")
