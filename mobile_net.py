# Only 2 lines will be added
# Rest of the flow and code remains the same as default keras
import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
import time
from checkpoint import create_checkpoint_callback
from load_data import load_dataset
import keras
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/training'
ds, image_count, label_names, steps_per_epoch = load_dataset(DATA_PATH, 32)

print(ds)
print(label_names)
print(image_count)
# quit()
mobile_net = keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), weights='imagenet', include_top=False)
mobile_net.trainable = False

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = create_checkpoint_callback(checkpoint_dir)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(image_count, 192, 192, 3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, (3, 3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, (3, 3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(label_names)),
])

model.compile(optimizer='rmsprop',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()

# log results

TEST_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/validation'
test_ds, test_image_count, test_label_names, test_steps_per_epoch = load_dataset(
    TEST_DATA_PATH, 32)

def change_range(image, label):
  return 2*image-1, label


keras_ds = ds.map(change_range)

model.fit(keras_ds, epochs=1, steps_per_epoch=int(
    steps_per_epoch), callbacks=[cp_callback], validation_data=np.array(test_ds))

# model.evaluate(np.array(test_ds), batch_size=32, steps=30)
