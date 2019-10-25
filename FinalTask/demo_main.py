import cv2
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib

def main():
    data_root = pathlib.Path("datasets/flower_photos")
    for item in data_root.iterdir():
        print(item)
    img = tf.io.read_file("./datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    img /= 255.0
    pass


if __name__ == "__main__()":
    main()
