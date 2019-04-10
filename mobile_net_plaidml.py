import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from load_data import load_dataset
from checkpoint import create_checkpoint_callback
import time

DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/training'
ds, image_count, label_names, steps_per_epoch = load_dataset(DATA_PATH, 32)


mobile_net = keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), weights='imagenet', include_top=False)
mobile_net.trainable = False

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = create_checkpoint_callback(checkpoint_dir)

model = keras.Sequential([
    keras.layers.Dense(300, input_shape=(192,192,3), activation='relu'),
    keras.layers.Dense(150, activation='relu'),
    # keras.layers.GlobalAveragePooling2D(),
    # keras.layers.Dense(len(label_names), activation='relu'),
    keras.layers.Activation('softmax')
])

model.compile(optimizer='sgd',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()

# log results

TEST_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/validation'
test_ds, test_image_count, test_label_names, test_steps_per_epoch = load_dataset(TEST_DATA_PATH, 32)

import numpy as np

model.fit(np.array(ds), epochs=1, steps_per_epoch=int(steps_per_epoch), callbacks=[cp_callback])

model.evaluate(np.array(test_ds), batch_size=32, steps=30)



