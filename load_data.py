from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pathlib
import random

tf.enable_eager_execution()

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    print(image)
    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def load_dataset(path, batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    DATA_PATH = path
    BATCH_SIZE = batch_size

    data_root = pathlib.Path(DATA_PATH)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)

    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())

    label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(
        path).parent.name] for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)

    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    
    steps_per_epoch = tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
    
    return  ds, image_count, label_names, steps_per_epoch