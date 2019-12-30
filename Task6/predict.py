import tensorflow as tf
import numpy as np
from main import get_data_v4
from vis import build_colormap2label
import matplotlib.pyplot as plt

keras = tf.keras
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_x(x, y):
    return x


def get_y(x, y):
    return y


def test_model(img_n=50):
    # color2label =
    model = keras.models.load_model('./models/val/first_model.h5', compile=False)

    data, n = get_data_v4()
    data = data.take(img_n)

    x = data.map(get_x, num_parallel_calls=AUTOTUNE)
    y_true = data.map(get_y, num_parallel_calls=AUTOTUNE)

    y_pred = model.predict(x.batch(1))
    y_pred = np.argmax(y_pred, axis=-1)
    pass


if __name__ == '__main__':
    test_model()



