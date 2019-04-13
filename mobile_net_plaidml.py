# # Only 2 lines will be added
# # Rest of the flow and code remains the same as default keras
# import plaidml.keras

# plaidml.keras.install_backend()

import tensorflow as tf
import os
# import keras
import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from load_data import load_dataset
# from checkpoint import create_checkpoint_callback
import time
import numpy as np


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# mobile_net = keras.applications.MobileNetV2(
#     input_shape=(192, 192, 3), weights='imagenet', include_top=False)
# mobile_net.trainable = False

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = create_checkpoint_callback(checkpoint_dir)

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(192, 192, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))


model.compile(optimizer='sgd',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    # this is the target directory
    '/Users/aboga/repos/car-damage-dataset/data2a/training',
    target_size=(192, 192),  # all images will be resized to 192x192
    batch_size=batch_size,
    class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

# train_generator = tf.cast(train_generator, tf.int64)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    '/Users/aboga/repos/car-damage-dataset/data2a/validation',
    target_size=(192, 192),
    batch_size=batch_size,
    class_mode='categorical')

# train_generator = tf.cast(validation_generator, tf.int64)

print('======== generated ========')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)
model.save_weights('first_try.h5')
