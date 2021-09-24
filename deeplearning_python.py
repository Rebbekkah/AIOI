# -*- coding: utf-8 -*-
"""DeepLearning_python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D0WLeI11GbqAg_lzunzJSwxDWM4A4NYF
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# model / data parameters
num_class = 10

# the data, split between train and tests sets
(xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

# scale images to the [0, 1] range in 32 bits
xtrain = xtrain.astype("float32")/255
xtest = xtest.astype("float32")/255

# make sure images have shape (28, 28, 1)
xtrain = np.expand_dims(xtrain, -1)
xtest = np.expand_dims(xtest, -1)

print("xrtain  shape : ", xtrain.shape)
print("xrtain : ", str(xtrain.shape))
print("yrtain : ", str(ytrain.shape))
print("xrtain : ", str(xtrain.shape))
print("xtest : ", str(xtest.shape))
print("ytest : ", str(ytest.shape))

# convert class vectors to binary class matrices
# ytrain = keras.utils.to_categorical(ytrain, num_class)
# ytest = keras.utils.to_categorical(ytest, num_class)

# plot image

from matplotlib import pyplot
for i in range(9) :
  pyplot.subplot(330 + 1 + i)
  pyplot.imshow(xtrain[i], cmap = pyplot.get_cmap('gray'))
pyplot.show()

xtrain = xtrain.reshape(60000, 784).astype("float32")/255
xtest = xtest.reshape(10000, 784).astype("float32")/255

input_shape = (784,)
inputs = keras.Input(shape = input_shape)
print(inputs.shape)
print(inputs.dtype)

dense = layers.Dense(64, activation = "relu")
x = dense(inputs)
x = layers.Dense(64, activation = "relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs = inputs, outputs = outputs, name = "mnist_model")

model.summary

keras.utils.plot_model(model, "my_first_model.png")

keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes = True)

# "compile" model
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
              optimizer = keras.optimizers.RMSprop(), metrics = ["accuracy"])

# keep history and fit model
history = model.fit(xtrain, ytrain, batch_size = 64, epochs = 5, validation_split = 0.2)

_, acc = model.evaluate(xtest, ytest, verbose = 1)
print('>%.3f' % (acc * 100.0))

# list all data in history
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()
