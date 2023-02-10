import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
import imghdr
from keras.applications.inception_v3 import InceptionV3, preprocess_input

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
                img_array = cv2.imread(os.path.join(os.path.join(path,p),file))
                new_img_array = cv2.resize(img_array, dsize=(200, 200))
                x.append(new_img_array)
                y.append(p)


create_test_data("photos")

x=np.array(x).reshape(-1, 200, 200, 3)
x=x/255.0
y=np.array(y)
y=labelencoder.fit_transform(y)

x_test,x_train,y_test,y_train = train_test_split(x,y,train_size=0.3,random_state=1)
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32')

i_model = InceptionV3(weights= 'imagenet', include_top=False, input_shape=(200, 200, 3))
#print(x_train.shape)
#print(x_test.shape)
model = Sequential()
model.add(i_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(12, activation = 'softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics = ['accuracy'])

checkpoint = "checkpoint-v3/v3-weights.ckpt"
if os.path.exists(checkpoint + ".index"):
    print("-----load_weights-----")
    model.load_weights(checkpoint)

cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,save_weights_only=True,save_best=True)

history = model.fit(x_train, y_train, epochs = 1, batch_size = 32, validation_data = (x_test, y_test),callbacks=[cp_callback])
model.summary()
