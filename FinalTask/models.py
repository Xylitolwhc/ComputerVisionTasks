from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import Model


class EmotionModel(Model):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.conv1 = Conv2D(32, 3, (1, 1), activation='relu', name='conv1')
        self.pool1 = AveragePooling2D((3, 3), name='pool1')
        self.conv2 = Conv2D(32, 3, (1, 1), activation='relu', name='conv2')
        self.pool1 = AveragePooling2D((3, 3), name='pool2')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu', name='dense1')
        self.d2 = Dense(7, activation='softmax', name='output')

