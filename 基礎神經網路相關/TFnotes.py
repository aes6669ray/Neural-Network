import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

def cost():
    y = Weights*x_data + biases
    loss = tf.reduce_mean(tf.square(y-y_data))
    return loss

optimizer = tf.optimizers.SGD(0.5)

for step in range(201):
    train = optimizer.minimize(cost, var_list=[Weights, biases])
    if step % 20 == 0:
        print(step, Weights.numpy(), biases.numpy())


# plot the real data
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

# for i in range(1000):
#     # training
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         # to visualize the result and improvement
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         prediction_value = sess.run(prediction, feed_dict={xs: x_data})
#         # plot the prediction
#         lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
#         plt.pause(0.1)