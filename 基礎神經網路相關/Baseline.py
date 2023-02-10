import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten
from tensorflow.keras import Model

cafir10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cafir10.load_data()
x_train = x_train/255
x_test = x_test/255


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5,5), padding="same")
        self.b1 = BatchNormalization()
        self.a1 = Activation("relu")
        self.p1 = MaxPool2D(pool_size=(2,2), strides=1, padding="same")
        self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation="relu")
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

model=Baseline()


model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["sparse_categorical_accuracy"]) 

model.fit(x_train,y_train,batch_size=32,epochs=3,validation_data=(x_test,y_test),validation_freq=1)

model.summary()