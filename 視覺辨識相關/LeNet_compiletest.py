import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import Model
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
cafir10 = tf.keras.datasets.cifar10
#tf.debugging.set_log_device_placement(True)


(x_train, y_train), (x_test, y_test) = cafir10.load_data()
x_train = x_train/255
x_test = x_test/255

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train/255
# x_test = x_test/255

# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)


model = Sequential()
model.add(Conv2D(filters = 6, kernel_size = (5, 5), padding = 'valid', strides=(1,1)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters = 16, kernel_size = (5, 5), strides=(1,1)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(120,activation = 'tanh'))
model.add(Dense(84,activation = 'tanh'))

model.add(Dense(10))
model.add(Activation('softmax'))


# model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), 
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
#               metrics = ['accuracy'])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics = ['accuracy'])

History = model.fit(x_train, y_train, epochs = 3, batch_size = 32, validation_data = (x_test, y_test))
#model.fit(x_train,y_train,batch_size=32,epochs=1,validation_data=(x_test,y_test),validation_freq=1)

model.summary()