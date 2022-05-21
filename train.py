import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

from evaluation import single_evaluate

# TEMP
n_hidden = 64
n_inputs = 8
n_outputs = 2
n_param = n_hidden*(n_inputs) + n_hidden + n_hidden * n_outputs + n_outputs


x_data = np.load('data/x_data.npy')
y_data = np.load('data/y_data.npy')

print(x_data.shape)

c = int(x_data.shape[0] * 0.8)

# NOTE! If we don't shuffle, we get extremely bad performance because we do not train on enough formations. 
# THIS IS A KEY FINDING.
# index = np.arange(x_data.shape[0])
# np.random.shuffle(index)

# x_data = x_data[index,:]
# y_data = y_data[index,:]

x_train = x_data[:c,:]
y_train = y_data[:c,:]



x_test = x_data[c:,:]
y_test = y_data[c:,:]


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(n_inputs,)))
model.add(tf.keras.layers.Dense(n_hidden, activation='tanh'))
# Now the model will take as input arrays of shape (None, 16)
# and output arrays of shape (None, 32).
# Note that after the first layer, you don't need to specify
# the size of the input anymore:
model.add(tf.keras.layers.Dense(n_outputs, activation='tanh'))

print(x_train.shape)


model.compile(optimizer='adam',loss='mse')


class CustomCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        weights = model.get_weights() 

        w1 = weights[0]
        b1 = weights[1]
        w2 = weights[2]
        b2 = weights[3]

        population = np.concatenate((w1.flatten(),b1.flatten(),w2.flatten(),b2.flatten()),axis=0)
        population = np.reshape(population,(1,n_param))

        fitness = single_evaluate(population)
        np.save('data/weights.npy', population)
        print(fitness)





print('Training starting')
model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=1000,verbose=1,callbacks=[CustomCallback()])
