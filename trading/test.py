from tensorflow.keras import layers, models, losses
import numpy as np
import tensorflow as tf
import keras
from tensorflow import math
tf.config.experimental_run_functions_eagerly(True)


def loss_fcn(y_true, y_pred, w):
    # loss = w * losses.mse(y_true, y_pred)
    loss = tf.math.greater_equal(y_true,y_pred)
    loss = tf.cast(loss,dtype=tf.float32)
    zero = tf.constant([0.0],dtype=tf.float32)
    added = tf.math.add(loss,zero)
    print(added)
    # if(a):
    #     return a
    # else:
    #     return a
    return loss


data_x = np.random.rand(5, 4, 1)
data_w = np.random.rand(5, 4)
data_y = np.random.rand(5, 4, 1)

x = layers.Input([4, 1])
y_true = layers.Input([4, 1])
w = layers.Input([4])
y = layers.Activation('tanh')(x)


model = models.Model(inputs=[x, y_true, w], outputs=y)
model.add_loss(loss_fcn(y, y_true, w))


model.compile()
model.fit((data_x, data_y, data_w))
print('all done')