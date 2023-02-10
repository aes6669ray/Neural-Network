import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# # y_pred = W*X + b
# W = tf.Variable(0.0)
# b = tf.Variable(0.0)

# # 定義損失函數
# def loss(y, y_pred):
#     return tf.reduce_mean(tf.square(y - y_pred))

# # 定義預測值
# def predict(X):
#     return W * X + b
    
# # 定義訓練函數
# def train(X, y, epochs=40, lr=0.0001):
#     current_loss=0
#     # 執行訓練
#     for epoch in range(epochs):
#         with tf.GradientTape() as t:
#             t.watch(tf.constant(X))
#             current_loss = loss(y, predict(X))

#         # 取得 W, b 個別的梯度
#         dW, db = t.gradient(current_loss, [W, b])
        
#         # 更新權重
#         # 新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
#         W.assign_sub(lr * dW) # W -= lr * dW
#         b.assign_sub(lr * db)

#         # 顯示每一訓練週期的損失函數
#         print(f'Epoch {epoch}: Loss: {current_loss.numpy()}') 


# # 產生隨機資料
# # random linear data: 100 between 0-50
# n = 100
# X = np.linspace(0, 50, n) 
# y = np.linspace(0, 50, n) 
  
# # Adding noise to the random linear data 
# X += np.random.uniform(-10, 10, n) 
# y += np.random.uniform(-10, 10, n) 

# # reset W,b
# W = tf.Variable(0.0)
# b = tf.Variable(0.0)

# # 執行訓練
# train(X, y)

# # W、b 的最佳解
# print(W.numpy(), b.numpy())

# plt.scatter(X, y, label='data')
# plt.plot(X, predict(X), 'r-', label='predicted')
# plt.legend()

# plt.show()


#============
#tf與keras八股

#import tensorflow as tf
#import numpy as np
from sklearn import datasets
import os

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(15)
np.random.shuffle(x_train)
np.random.seed(15)
np.random.shuffle(y_train)
tf.random.set_seed(15)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["sparse_categorical_accuracy"]) 

checkpoint = "checkpoint/test.ckpt"
if os.path.exists(checkpoint + ".index"):
    print("-----load_weights-----")
    model.load_weights(checkpoint)

cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,save_weights_only=True,save_best=True)

model.fit(x_train,y_train,batch_size=32,epochs=10,validation_split=0.2,validation_freq=20,callbacks=[cp_callback])

model.summary()

