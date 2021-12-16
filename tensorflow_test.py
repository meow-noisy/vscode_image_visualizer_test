# 参考
# https://www.tensorflow.org/tutorials/load_data/images?hl=ja

import tensorflow as tf
import matplotlib.pyplot as plt


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


im_filepath = 'pexels-oswald-elsaboath-7061955_720x480.jpg'

img_raw = tf.io.read_file(im_filepath)

print(repr(img_raw)[:100] + "...")

img_tensor = tf.image.decode_image(img_raw)


im = load_and_preprocess_image(im_filepath)

plt.imshow(im)

print(img_tensor.shape)
print(img_tensor.dtype)
