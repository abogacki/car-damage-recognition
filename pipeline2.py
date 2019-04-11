from __future__ import absolute_import, division, print_function
import pathlib
import random
import tensorflow as tf

tf.enable_eager_execution()
tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE


DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/training'
data_root = pathlib.Path(DATA_PATH)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

label_names = sorted(
    item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(
    path).parent.name] for path in all_image_paths]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))


def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)

BATCH_SIZE = 32

ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)

ds.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), weights='imagenet', include_top=False)
mobile_net.trainable = False

res_net = tf.keras.applications.ResNet50(
    input_shape=(192, 192, 3), weights=None, include_top=False, pooling='avg')
res_net.trainable = True


def change_range(image, label):
  return 2*image-1, label


keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = res_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(192, 192, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_names)),
    tf.keras.layers.Activation('softmax')
])


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()


steps_per_epoch = int(tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy())


from tensorflow.keras.callbacks import TensorBoard

import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

model.fit(keras_ds, epochs=10, batch_size=BATCH_SIZE, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard], verbose=1)




