from model.convNeXt_2020s import *
from model.resnet50_resnet50Xt import ResNeXt, ResNet
from data import load_dataset_original, load_dataset_cifar10,cutmix,preprocess_image
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
# pip install keras-adamw
from keras_adamw import AdamW
from tensorflow.python.data import Dataset

import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument('--model', default='convnext', type=str,
                        help='Type of ConvNeXt model, valid option: resnet50, resnext')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
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
    AUTO = tf.data.AUTOTUNE

    # Data loader
    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds, val_ds = load_dataset_original(train_folder,valid_folder,batch_size)

    else:
        print("Data folder is not set. Use CIFAR 10 dataset")
        train_ds_cmu, train_ds_simple, val_ds = load_dataset_cifar10(batch_size)


    # if args.model == 'resnet50':
    #     model = ResNet(image_size, image_size, image_channels, num_filters, problem_type=problem_type, onum_classes=num_classes, pooling='avg', dropout_rate=False).ResNet50()
    # elif args.model == 'resnext':
    #     model = ResNeXt(image_size, image_size, image_channels, num_filters,cardinality=32, problem_type=problem_type, onum_classes=num_classes, pooling='avg', dropout_rate=False).ResNetXt50()
    # else:
    model = convnext(
        input_shape=[224,224,3],
        classes = args.num_classes,
    )

    #optimizer= AdamW(learning_rate=args.lr)
    optimizer = Adam(learning_rate=args.lr)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])
    # model.summary()
    # Traning
    
    model.fit(train_ds_cmu,
              epochs=10,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)

