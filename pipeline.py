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
label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

img_path = all_image_paths[0]
img_raw = tf.read_file(img_path)

img_tensor = tf.image.decode_image(img_raw)
img_final = tf.image.resize_images(img_tensor, [192, 192])
img_final = img_final/255.0


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0 # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print('image shape: ', image_label_ds.output_shapes[0])
print('label shape: ', image_label_ds.output_shapes[1])
print('types: ', image_label_ds.output_types)
print()
print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label) 

BATCH_SIZE = 32

ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)

ds.prefetch(buffer_size=AUTOTUNE)

ds

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3), include_top=False)
mobile_net.trainable=False

def change_range(image, label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape) 

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))
])

logit_batch = model(image_batch).numpy()

print("min logit: ", logit_batch.min())
print("max logit: ", logit_batch.max())
print()

print('Shape: ', logit_batch.shape)

model.compile(optimizer=tf.train.AdamOptimizer(),
loss=tf.keras.losses.sparse_categorical_crossentropy,
metrics=["accuracy"])

model.summary()

steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(ds, epochs=1, steps_per_epoch=int(steps_per_epoch))