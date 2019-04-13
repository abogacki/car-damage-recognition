from PIL import Image
import numpy as np
import os
import imageio
import pathlib
import random
import keras

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

TRAIN_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/training'
TEST_DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/validation'

all_train_paths = pathlib.Path(TRAIN_DATA_PATH)
all_test_paths = pathlib.Path(TEST_DATA_PATH)


train_image_paths = list(all_train_paths.glob('*/*'))
train_image_paths = [str(path) for path in train_image_paths]

test_image_paths = list(all_test_paths.glob('*/*'))
test_image_paths = [str(path) for path in test_image_paths]

random.shuffle(train_image_paths)
random.shuffle(test_image_paths)

label_names = sorted(
    item.name for item in all_train_paths.glob('*/') if item.is_dir())
label_to_index = dict((name, index)
                      for index, name in enumerate(label_names))

train_image_labels = [label_to_index[pathlib.Path(
    path).parent.name] for path in train_image_paths]
test_image_labels = [label_to_index[pathlib.Path(
    path).parent.name] for path in test_image_paths]




def preprocess_image(image, size=(192, 192), conv_type=float):
    image = Image.open(image)
    image = image.resize(size)  # resize 192x192
    image = np.asarray(image).astype(conv_type)  # convert to numpy array
    image /= 255.0  # normalize [0,1] values
    return image


def attach_label_to_image(image, label_index):
    print(label_index)


train_normalized_images = np.array(
    [preprocess_image(image) for image in train_image_paths])
test_normalized_images = np.array(
    [preprocess_image(image) for image in test_image_paths])

print(train_normalized_images.shape)

# quit()

model = keras.Sequential()
model.add(keras.layers.Convolution2D(32, (3, 3), input_shape=(192, 192, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(3))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

print("Length normalized images", len(train_normalized_images))
print("Length normalized images", len(train_image_labels))
print("Shape normalized images", train_normalized_images.shape)
print("Shape normalized labels", train_image_labels)

model.fit(train_normalized_images, train_image_labels, batch_size=30, epochs=2, verbose=1)

loss, acc = model.evaluate(test_normalized_images,
                           test_image_labels, verbose=1)

