import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from load_data import load_dataset
from checkpoint import create_checkpoint_callback
import time

tf.enable_eager_execution()


DATA_PATH = '/Users/aboga/repos/car-damage-dataset/data2a/training'
ds, image_count, label_names, steps_per_epoch = load_dataset(DATA_PATH, 32)

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds.prefetch(buffer_size=AUTOTUNE)


mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), weights=None, include_top=False)
mobile_net.trainable = False 

path = '/Users/aboga/repos/car-classification/mobile_net/weights/01'
cp_callback = create_checkpoint_callback(path)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='relu'),
    tf.keras.layers.Activation('softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.summary()

# log results
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))


model.fit(ds, epochs=1, steps_per_epoch=int(steps_per_epoch), callbacks=[cp_callback, tensorboard])
