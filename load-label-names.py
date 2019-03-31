from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()
tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data1a/training/'
data_root = pathlib.Path(DATA_PATH)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)