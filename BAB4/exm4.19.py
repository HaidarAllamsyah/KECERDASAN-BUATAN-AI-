#Example 4.19 SimpleRNN2.py
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, Input    # [FIX]
from keras import optimizers

model = Sequential()
model.add(Input(shape=(49, 1)))                                 # [FIX]
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(46))
model.add(Activation('softmax'))
adam = optimizers.Adam(learning_rate=0.001)                     # [FIX] lr -> learning_rate
model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])
print(model.summary())
