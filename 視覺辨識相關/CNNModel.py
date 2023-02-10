import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import imghdr

labelencoder = LabelEncoder()


x=[]
y=[]
def create_test_data(path):
    for p in os.listdir(path):
        for i,file in enumerate(os.listdir(os.path.join(path,p))):
            if imghdr.what(os.path.join(os.path.join(path,p),file)) != "jpeg":
                os.remove(file)
                print("del",file)
            else:
                img_array = cv2.imread(os.path.join(os.path.join(path,p),file),cv2.IMREAD_GRAYSCALE)
                new_img_array = cv2.resize(img_array, dsize=(200, 200))
                x.append(new_img_array)
                y.append(p)


create_test_data("photos")

x=np.array(x).reshape(-1, 200, 200, 1)
x=x/255.0
y=np.array(y)
y=labelencoder.fit_transform(y)

x_test,x_train,y_test,y_train = train_test_split(x,y,train_size=0.3,random_state=1)
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32')

#print(y_train.shape)
#print(y_test.shape)
model = Sequential()

model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = x.shape[1:], kernel_regularizer=tf.keras.regularizers.l2()))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
Dropout(0.2)
model.add(Dense(12, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_data = (x_test, y_test))
model.summary()
