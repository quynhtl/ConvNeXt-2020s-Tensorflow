from model.convNet import *
from model.resnet50_resnet50Xt import ResNeXt, ResNet
from data import *
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
import numpy as np

import os
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.data import Dataset
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument('--model', default='macro', type=str,
                        help='Type of ConvNeXt model, valid option: resnet50, resnext')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument('--num-classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--num-filters', default=16,
    type=int, help='Number of filters')
    parser.add_argument('--image-size', default=224,
                        type=int, help='Size of input image')
    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')
    parser.add_argument('--train-folder', default='', type=str,
                        help='Where training data is located')
    parser.add_argument('--valid-folder', default='', type=str,
                        help='Where validation data is located')
    parser.add_argument('--class-mode', default='categorical', type=str, help='Class mode to compile')
    parser.add_argument('--problem-type', default='Classification', type=str)
    parser.add_argument('--model-folder', default='output/',
                        type=str, help='Folder to save trained model')  
    parser.add_argument('--cardinality', default=32,
                        type=int, help='cardinality')
    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ConvNext-2020s paper Implementation-------------------')
    print('---------------------------------------------------------------------')
    print('Training ConvNext2020s model with hyper-params:') 
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Assign arguments to variables to avoid repetition 
    train_folder = args.train_folder
    valid_folder = args.valid_folder
    batch_size =  args.batch_size
    image_size = args.image_size
    image_channels = args.image_channels
    num_classes = args.num_classes
    epoch = args.epochs
    class_mode = args.class_mode
    lr = args.lr
    num_filters = args.num_filters
    problem_type = args.problem_type
    cardinality= args.cardinality


    # Data loader
    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds_mu = image_dataset_from_directory(
            args.train_folder,
            seed=123,
            image_size=(image_size, image_size),
            shuffle=True,
            batch_size=batch_size,
        )
        val_ds = image_dataset_from_directory(
            args.valid_folder,
            seed=123,
            image_size=(image_size, image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )

    else:
        print("Data folder is not set. Use CIFAR 10 dataset")

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        AUTO = tf.data.AUTOTUNE
        BATCH_SIZE = 32
        IMG_SIZE = 224

        def preprocess_image(image, label):
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
            return image, label

        val_samples = 2000
        x_val, y_val = x_train[:val_samples], y_train[:val_samples]
        new_x_train, new_y_train = x_train[val_samples:], y_train[val_samples:]

        train_ds_one = (
            tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
            .shuffle(BATCH_SIZE * 100)
            .batch(BATCH_SIZE)
        )
        train_ds_two = (
            tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
            .shuffle(BATCH_SIZE * 100)
            .batch(BATCH_SIZE)
        )
        # Because we will be mixing up the images and their corresponding labels, we will be
        # combining two shuffled datasets from the same training data.
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
        train_ds_mu = train_ds.map(
            lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
        )
    if args.model == 'resnet50':
        model = ResNet(image_size, image_size, image_channels, num_filters, problem_type=problem_type, onum_classes=num_classes, pooling='avg', dropout_rate=False).ResNet50()
    elif args.model == 'resnext':
        model = ResNeXt(image_size, image_size, image_channels, num_filters,cardinality=32, problem_type=problem_type, onum_classes=num_classes, pooling='avg', dropout_rate=False).ResNetXt50()
    else:
        model = convnext(
            input_shape=(image_size,
                             image_size, image_channels),
            num_classes = num_classes,
            image_size = image_size
        )

    model.build(input_shape=(None, image_size,
                             image_size, image_channels))

    optimizer= tf.keras.optimizers.experimental.AdamW(learning_rate=args.lr)

    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    # Traning
    model.fit(train_ds_mu,
              epochs=args.epochs,
              batch_size=batch_size,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)

