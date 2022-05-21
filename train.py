import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


x_data = np.load('data/x_data.npy')
y_data = np.load('data/y_data.npy')

print(x_data.shape)

c = int(x_data.shape[0] * 0.8)

# NOTE! If we don't shuffle, we get extremely bad performance because we do not train on enough formations. 
# THIS IS A KEY FINDING.
index = np.arange(x_data.shape[0])
np.random.shuffle(index)

x_data = x_data[index,:]
y_data = y_data[index,:]

x_train = x_data[:c,:]
y_train = y_data[:c,:]



x_test = x_data[c:,:]
y_test = y_data[c:,:]


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(8,)))
model.add(tf.keras.layers.Dense(32, activation='tanh'))
# Now the model will take as input arrays of shape (None, 16)
# and output arrays of shape (None, 32).
# Note that after the first layer, you don't need to specify
# the size of the input anymore:
model.add(tf.keras.layers.Dense(2, activation='tanh'))

print(x_train.shape)


model.compile(optimizer='adam',loss='mse')

print('Training starting')
model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=100,verbose=1)
