import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras_preprocessing.image import ImageDataGenerator
np.random.seed(42)
tf.random.set_seed(42)
AUTO = tf.data.AUTOTUNE

image_size = 224
def preprocess_image(image,label):
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = image_size  * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = image_size  * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=image_size , dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=image_size , dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, image_size )
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, image_size )
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, image_size )
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, image_size )

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, image_size, image_size
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, image_size, image_size
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (image_size * image_size)
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label

def load_dataset_cifar10(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    train_ds_one = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
    )
    train_ds_two = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
    )

    train_ds_simple = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds_simple = (
        train_ds_simple.map(preprocess_image, num_parallel_calls=AUTO)
        .batch(batch_size)
        .prefetch(AUTO)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = (
        val_ds.map(preprocess_image, num_parallel_calls=AUTO)
        .batch(batch_size)
        .prefetch(AUTO)
    )

    # Combine two shuffled datasets from the same training data.
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two)) 
    train_ds_cmu = (
        train_ds.shuffle(1024)
        .map(cutmix, num_parallel_calls=AUTO)
        .batch(batch_size)
        .prefetch(AUTO)
    )

    return train_ds_cmu, train_ds_simple, val_ds


def load_dataset_original(train_folder,valid_folder,batch_size):
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
        seed=123,
        image_size=(image_size, image_size),
        shuffle=True,
        batch_size=batch_size,
    )
    val_ds = val_datagen.flow_from_directory(
        valid_folder,
        seed=123,
        image_size=(image_size, image_size),
        shuffle=True,
        batch_size=batch_size,
    )
    return train_ds, val_ds

