
import tensorflow as tf
import os

# Create checkpoint callback
def create_checkpoint_callback(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)

    return tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)
