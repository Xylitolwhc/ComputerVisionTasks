import tensorflow as tf
import numpy as np

from transform import voc_label_indices, build_colormap2label

keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE

x_dir = './VOCdevkit/VOC2012/img/'
y_dir = './VOCdevkit/VOC2012/SegmentationClass/'


# Input utilities
def voc_label_indices_for_tf(y, colormap2label):
    """Map a RGB color image to a label."""

    def fn(val):
        return tf.map_fn(lambda x: colormap2label[x], val, back_prop=False)

    y = tf.cast(y, tf.int32)
    idx = ((y[:, :, 0] * 256 + y[:, :, 1]) * 256
           + y[:, :, 2])
    return tf.map_fn(fn, idx, back_prop=False)


def preprocess_with_random_crop(aug_n=10):
    def _preprocess(name):
        x = tf.strings.join([x_dir, name, '.jpg'])
        y = tf.strings.join([y_dir, name, '.png'])

        x = process_path(x)
        y = process_path(y)

        x = keras.applications.vgg16.preprocess_input(tf.cast(x, tf.float32))
        y = voc_label_indices_for_tf(y, tf.convert_to_tensor(build_colormap2label()))[..., tf.newaxis]
        y = tf.cast(y, tf.float32)

        img = tf.concat([x, y], axis=2)
        rst = tf.image.random_crop(img, (224, 224, 4))
        aug_x = [rst[:, :, :-1]]
        aug_y = [rst[:, :, -1][..., tf.newaxis]]

        for _ in range(aug_n - 1):
            rst = tf.image.random_crop(img, (224, 224, 4))
            aug_x.append(rst[:, :, :-1])
            aug_y.append(rst[:, :, -1][..., tf.newaxis])

        return tf.data.Dataset.from_tensor_slices((aug_x, aug_y))

    return _preprocess


def preprocess(name):
    x = tf.strings.join([x_dir, name, '.jpg'])
    y = tf.strings.join([y_dir, name, '.png'])

    x = process_path(x)
    y = process_path(y)

    x = preprocess_img(x)
    y = voc_label_indices_for_tf(y, tf.convert_to_tensor(build_colormap2label()))
    y = tf.image.resize_with_crop_or_pad(y[..., np.newaxis], 512, 512)

    return x, y


def process_name(name=''):
    x = process_path(tf.strings.join([x_dir, name + '.jpg']))
    x = preprocess_img(x)
    y = np.array(keras.preprocessing.image.load_img(tf.strings.join([y_dir, name + '.png'])))
    y = voc_label_indices(y, build_colormap2label()).astype(np.int32)
    y = pad_and_crop(y)
    return x, y


def pad_and_crop(img):
    if img.ndim == 3:
        return tf.image.resize_with_crop_or_pad(img, 512, 512)
    elif img.ndim == 2:
        return tf.reshape(tf.image.resize_with_crop_or_pad(img[..., np.newaxis], 512, 512), [512, 512])


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def preprocess_img(img):
    img = tf.cast(img, tf.float32)
    img = keras.applications.vgg16.preprocess_input(img)
    return tf.image.resize_with_crop_or_pad(img, 512, 512)


# Construct input pipeline
def get_data():
    trainval_file_names = np.loadtxt(
        './VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)

    data = tf.data.Dataset.from_tensor_slices(trainval_file_names).shuffle(1000)
    data = data.map(preprocess, AUTOTUNE)
    return data, len(trainval_file_names)


def get_data_random_crop(aug_n=10):
    trainval_file_names = np.loadtxt(
        './VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', dtype=np.str)
    data = tf.data.Dataset.from_tensor_slices(trainval_file_names).shuffle(1000)

    data = data.interleave(preprocess_with_random_crop(aug_n), cycle_length=4, num_parallel_calls=AUTOTUNE)

    return data, len(trainval_file_names) * aug_n
