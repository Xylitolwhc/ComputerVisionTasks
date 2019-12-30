import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import *


def train_emotion_model():
    fer2013 = pd.read_csv("datasets/fer2013/fer2013.csv")

    train_data = fer2013[fer2013['Usage'] == "Training"]
    public_test_data = fer2013[fer2013['Usage'] == "PublicTest"]
    private_test_data = fer2013[fer2013['Usage'] == "PrivateTest"]

    train = train_data.reset_index().drop(['Usage', 'index'], axis=1)
    test = public_test_data.reset_index().drop(['Usage', 'index'], axis=1)

    train_data, train_label = get_data(train)
    test_data, test_label = get_data(test)

    model = get_emotion_model()
    model.fit(train_data, train_label, batch_size=32, shuffle=10000, epochs=5)
    model.evaluate(test_data, test_label)
    pass


def get_data(data):
    x = []
    y = []
    for index, data in data.iterrows():
        emotion = data['emotion']
        pixels = data['pixels']
        pixels = np.asarray([float(p) for p in pixels.split()]).reshape(48, 48)

        x.append(pixels)
        y.append(emotion)

    x = np.asarray(x)
    x = np.expand_dims(x, axis=3)
    y = np.asarray(y)

    return x, y


def get_emotion_model():
    model = tf.keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', name='conv1'),
        MaxPool2D((2, 2), 1, name="pool1"),
        Conv2D(32, (3, 3), activation='relu', name='conv2'),
        MaxPool2D((2, 2), 1, name="pool2"),
        Flatten(),
        Dense(128, activation='relu', name='dense1'),
        Dense(7, activation='softmax', name='output')]
    )
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_emotion_model()
