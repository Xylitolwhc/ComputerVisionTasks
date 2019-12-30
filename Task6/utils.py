import numpy as np
import tensorflow as tf

keras = tf.keras
K = keras.backend


# Custom keras objects

# Compare performances
def voc_seg_acc(y_true, y_pred):
    # Note y_true and y_pred always have the same shape, and x and y are automatically converted to the same ndim
    y_pred = K.argmax(y_pred, axis=-1)[..., tf.newaxis]
    y_true = K.cast(y_true, tf.int64)
    rst = K.equal(y_pred, y_true)
    return K.mean(rst)


def voc_seg_acc_v2(y_true, y_pred):  # Note y_true and y_pred always have the same shape, and x and y are automatically converted to the same ndim
    y_pred = K.reshape(y_pred, (-1, 21))
    y_true = K.reshape(y_true, (-1, 1))

    return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def voc_seg_loss(y_true, y_pred):
    # shape = K.shape(y_true)

    y_true = K.reshape(y_true, [-1, 1])
    y_pred = K.reshape(y_pred, [-1, 21])

    return K.sparse_categorical_crossentropy(y_true, y_pred)


def voc_seg_loss_v2(y_true, y_pred):
    y_true = K.one_hot(K.cast(K.reshape(y_true, (-1,)), tf.int32), 21)
    y_pred = K.reshape(y_pred, [-1, 21])

    rst = y_true * y_pred
    rst = K.sum(rst, axis=-1)
    return K.sum(K.log(rst))