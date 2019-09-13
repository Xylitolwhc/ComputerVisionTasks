import tensorflow as tf
from sklearn.datasets import fetch_mldata
import sys
import mnist
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift

batch_size = 128

sys.path.append('mnist')


def load_data():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    x_train_shift = []
    y_train_augmented = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(train_images, train_labels):
            x_train_shift.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)
    x_train_shift = np.array(x_train_shift)
    y_train_augmented = np.array(y_train_augmented)

    print(x_train_shift.shape, y_train_augmented.shape, test_images.shape, test_labels.shape)

    return x_train_shift, y_train_augmented, test_images, test_labels


def shuffle_bath(x, y, batch_size):
    rnd_idx = np.random.permutation(len(x))
    n_batches = len(x) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        x_batch, y_batch = x[batch_idx], y[batch_idx]
        yield x_batch, y_batch


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), name='reshape1'),
        tf.keras.layers.Conv2D(12, [3, 3], strides=1, padding='SAME', name='conv1',
                               input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, name='pool1'),
        tf.keras.layers.Conv2D(16, [3, 3], strides=1, padding='SAME', name='conv2'),
        tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, name='pool2'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu, name='fc1'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(X_train, Y_train, X_test, Y_test):
    tf.compat.v1.reset_default_graph()
    with tf.name_scope('placeholders'):
        X = tf.compat.v1.placeholder(np.float32, shape=[None, 28, 28], name='X')
        X_reshaped = tf.reshape(X, [-1, 28, 28, 1], name='X_reshaped')
        print(X_reshaped.shape)
        y = tf.compat.v1.placeholder(np.int32, shape=None, name='y')

    with tf.name_scope('conv'):
        conv1 = tf.layers.Conv2D(X_reshaped, filters=12, kernel_szie=[3, 3], strides=1, padding='SAME',
                                 name='conv1')
        pool1 = tf.layers.MaxPooling2D(conv1, [3, 3], strides=2, name='pool1')

        conv2 = tf.layers.Conv2D(pool1, 16, [3, 3], strides=1, padding='SAME', name='conv2')
        pool2 = tf.layers.MaxPooling2D(conv2, [3, 3], strides=2, name='pool2')

        pool2_flatten = tf.reshape(pool2, shape=(-1, 6 * 6 * 16))

        fc1 = tf.layers.Dense(pool2_flatten, 256, activation=tf.nn.relu, name='fc1')
        # fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2')

        logits = tf.layers.Dense(fc1, 10, activation=tf.nn.relu, name='output')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)

    with tf.name_scope('train'):
        optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, np.float32))

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        out = []
        for epoch in range(20):
            for x_batch, y_batch in shuffle_bath(X_train, Y_train, batch_size):
                session.run(training_op, feed_dict={X: x_batch, y: y_batch})
            if epoch % 1 == 0:
                batch_acc = accuracy.eval(feed_dict={X: x_batch, y: y_batch})
                val_acc = accuracy.eval(feed_dict={X: X_test, y: Y_test})
                print(epoch, "Batch Acc = ", batch_acc, "Validation Acc = ", val_acc)
                outputs = session.run(logits, feed_dict={X: X_test})
                out.append(outputs)


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dx, dy], cval=0, mode='constant')
    return shifted_image


def plot_image(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_error = {}
    n_error = []
    for i in range(len(y_pred)):
        if y_pred[i] != Y_test[i]:
            n_error.append(i)
            if Y_test[i] in y_error.keys():
                y_error[Y_test[i]] += 1
            else:
                y_error[Y_test[i]] = 1

    plt.figure(1, (10, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(4):
        for j in range(4):
            n = n_error[i * 2 + j]
            ax = plt.subplot(4, 4, i * 4 + j + 1)
            ax.set_title("pred:" + str(y_pred[n]) + " acc:" + str(Y_test[n]))
            ax.imshow(X_test[n])
    plt.show()


def __main__():
    X_train, Y_train, X_test, Y_test = load_data()

    model = create_model()

    model.fit(X_train, Y_train, epochs=2)

    test_loss, test_acc = model.evaluate(X_test, Y_test)

    print("Test Acc = ", test_acc)

    plot_image(model, X_test, Y_test)


if __name__ == '__main__':
    __main__()
