# # Only 2 lines will be added
# # Rest of the flow and code remains the same as default keras
import plaidml.keras

plaidml.keras.install_backend()

# Rest =====================

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

train_image_labels = [pathlib.Path(
    path).parent.name for path in train_image_paths]
train_image_labels = np.array(train_image_labels)

test_image_labels = [label_to_index[pathlib.Path(
    path).parent.name] for path in test_image_paths]
test_image_labels = np.array(test_image_labels)


def preprocess_image(image, size=(224, 224), conv_type=float):
    image = Image.open(image)
    image = image.resize(size)  # resize 192x192
    image = np.asarray(image).astype(conv_type)  # convert to numpy array
    # image = np.expand_dims(image, axis=0)
    image = keras.applications.resnet50.preprocess_input(image)
    return image


train_normalized_images = np.array(
    [preprocess_image(image) for image in train_image_paths])
test_normalized_images = np.array(
    [preprocess_image(image) for image in test_image_paths])

res_net = keras.applications.ResNet50(
    input_shape=(224, 224, 3), weights='imagenet', include_top=False)
res_net.trainable = True


model = keras.Sequential()
model.add(res_net)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(1024, activation="relu"))
model.add(keras.layers.Dense(1024, activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

train_label_encoder = LabelEncoder()
train_integer_encoded = train_label_encoder.fit_transform(train_image_labels)
train_onehot_encoder = OneHotEncoder(sparse=False)
train_integer_encoded = train_image_labels.reshape(len(train_image_labels), 1)
train_onehot_encoded = train_onehot_encoder.fit_transform(
    train_integer_encoded)

test_label_encoder = LabelEncoder()
test_integer_encoded = test_label_encoder.fit_transform(test_image_labels)
test_onehot_encoder = OneHotEncoder(sparse=False)
test_integer_encoded = test_image_labels.reshape(len(test_image_labels), 1)
test_onehot_encoded = test_onehot_encoder.fit_transform(
    test_integer_encoded)

print(len(test_normalized_images))
print(len(test_onehot_encoded))

model.fit(train_normalized_images, train_onehot_encoded,
          validation_data=(test_normalized_images, test_onehot_encoded), batch_size=30, epochs=10, verbose=1)

model.save_weights('./weights/res_net_weights.h5')

# loss, acc = model.evaluate(test_normalized_images,
#                            test_onehot_encoded, verbose=1)

# print(loss, acc)
