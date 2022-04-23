import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras_preprocessing.image import ImageDataGenerator
np.random.seed(42)
tf.random.set_seed(42)

AUTO = tf.data.AUTOTUNE
def load_dataset_cifar10(image_size):
    BATCH_SIZE = 32
    def preprocess_image(image,label):
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
        return image, label

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    train_ds_simple = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_ds_simple = (
        train_ds_simple.map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return train_ds_simple, val_ds

def load_dataset_original(train_folder,valid_folder,image_size):
    batch_size = 32
    train_datagen = ImageDataGenerator(rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_ds = train_datagen.flow_from_directory(
            train_folder,
            target_size=(image_size, image_size),
            batch_size= batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=123,
        )
    val_ds = val_datagen.flow_from_directory(
            valid_folder,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=123,
        )
    
    return train_ds,val_ds 

