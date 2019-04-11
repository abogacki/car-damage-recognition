from PIL import Image
import numpy as np
import os
import imageio
import pathlib
import random

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

label_names = sorted(item.name for item in all_train_paths.glob('*/') if item.is_dir())
label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))

train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
test_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_image_paths]



naming_dict = {}



def preprocess_image(image, size=(192,192), conv_type=float):
    image = Image.open(image) 
    image = image.resize(size) # resize 192x192
    image = np.asarray(image).astype(conv_type) # convert to numpy array
    image /= 255.0 # normalize [0,1] values
    return image


normalized_images = np.array([preprocess_image(image) for image in test_image_paths])

print(normalized_images)


quit()

