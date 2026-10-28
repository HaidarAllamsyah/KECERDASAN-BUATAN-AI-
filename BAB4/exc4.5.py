# Exercise 4.5 - AlexNet dengan jumlah unit Dense berbeda
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

def build_alexnet(d1=4096, d2=4096, d3=1000):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(d1, activation='relu')); model.add(Dropout(0.4))
    model.add(Dense(d2, activation='relu')); model.add(Dropout(0.4))
    model.add(Dense(d3, activation='relu')); model.add(Dropout(0.4))
    model.add(Dense(17, activation='softmax'))
    return model

for d in [(1024, 1024, 256), (2048, 2048, 500), (4096, 4096, 1000), (8192, 8192, 1000)]:
    print(f"Dense {d} -> total parameter = {build_alexnet(*d).count_params():,}")
