import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D, Add, Activation

keras = tf.keras
VGG16 = keras.applications.vgg16.VGG16


def fcn_8s(input_shape=(None, None, 3), output_classes=21, logits_output=True):
    """FCN_8s model using keras functional API, with weights transferred from vgg16."""
    inputs = keras.Input(shape=input_shape)

    block1_conv1 = Conv2D(64, [3, 3], padding='same', activation='relu', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, [3, 3], padding='same', activation='relu', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPool2D((2, 2), padding='same', name='block1_pool')(block1_conv2)

    block2_conv1 = Conv2D(128, [3, 3], padding='same', activation='relu', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, [3, 3], padding='same', activation='relu', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPool2D((2, 2), padding='same', name='block2_pool')(block2_conv2)

    block3_conv1 = Conv2D(256, [3, 3], padding='same', activation='relu', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, [3, 3], padding='same', activation='relu', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, [3, 3], padding='same', activation='relu', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPool2D((2, 2), padding='same', name='block3_pool')(block3_conv3)

    block4_conv1 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPool2D((2, 2), padding='same', name='block4_pool')(block4_conv3)

    block5_conv1 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, [3, 3], padding='same', activation='relu', name='block5_conv3')(block5_conv2)
    block5_pool = MaxPool2D((2, 2), padding='same', name='block5_pool')(block5_conv3)

    fc1 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(block5_pool)
    fc2 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(fc1)
    fc2_score = Conv2D(output_classes, (1, 1), padding='same', name='fc2_score')(fc2)
    fc2_4x = UpSampling2D(size=(4, 4), interpolation='bilinear', trainable=False, name='fc2_4x')(fc2_score)

    pool4_score = Conv2D(output_classes, (1, 1), padding='same', name='pool4_score')(block4_pool)
    pool4_2x = UpSampling2D((2, 2), interpolation='bilinear', trainable=False, name='pool4_2x')(pool4_score)

    pool3_score = Conv2D(output_classes, (1, 1), padding='same', name='poo3_score')(block3_pool)

    add = Add()([fc2_4x, pool4_2x, pool3_score])
    add_8x = UpSampling2D((8, 8), interpolation='bilinear', trainable=False, name='add_8x')(add)

    if logits_output is True:
        model = keras.Model(inputs=inputs, outputs=add_8x)
    elif logits_output is False:
        outputs = Activation('softmax', name='softmax')(add_8x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    else:
        raise ValueError('Only True or False  are supported for logits_output argument.')

    # transfer weights
    for layer in VGG16().layers:
        name = layer.name
        if 'block' in name:
            model.get_layer(name).set_weights(layer.get_weights())
        elif name == 'fc1':
            fc1_weights = layer.get_weights()
            fc1_weights[0] = fc1_weights[0].reshape((7, 7, 512, 4096))
            model.get_layer('fc1').set_weights(fc1_weights)
        elif name == 'fc2':
            fc2_weights = layer.get_weights()
            fc2_weights[0] = fc2_weights[0].reshape((1, 1, 4096, 4096))
            model.get_layer('fc2').set_weights(fc2_weights)

    # for layer in model.layers:
    #     if 'conv' in layer.name and int(layer.name[5]) <= 3:
    #         layer.trainable = False

    return model
