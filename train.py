from model.convNeXt_2020s import convnext
from model.resnet50_resnet50Xt import ResNeXt, ResNet
from data import load_dataset_original, load_dataset_cifar10
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
# conda deactivate (trong TH đang ở môi trg khác dùng lệnh này để thoát khỏi enviroment đó r mới cài thư viện dưới nhé)
# pip install tensorflow-addons
import tensorflow_addons as tfa
import os
from optimizer_adamW import WeightDecayScheduler, lr_schedule, wd_schedule
from tensorflow.python.data import Dataset
from keras_preprocessing.image import ImageDataGenerator



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
        train_datagen = ImageDataGenerator(rotation_range=15,
                                            rescale=1./255,
                                            shear_range=0.1,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1)
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        #Load train set
        train_ds_cmu = train_datagen.flow_from_directory(
            train_folder,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=123,
        )
        #Load test set
        val_ds = val_datagen.flow_from_directory(
            valid_folder,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=123,
        )

    else:
        print("Data folder is not set. Use CIFAR 10 dataset")
        train_ds_cmu, train_ds_simple, val_ds = load_dataset_cifar10(batch_size)


    if args.model == 'resnet50':
        model = ResNet(image_size, image_size, image_channels, num_filters, problem_type=problem_type, onum_classes=num_classes, pooling='avg', dropout_rate=False).ResNet50()
    elif args.model == 'resnext':
        model = ResNeXt(image_size, image_size, image_channels, num_filters,cardinality=32, problem_type=problem_type, onum_classes=num_classes, pooling='avg', dropout_rate=False).ResNetXt50()
    else:
        model = convnext(
            input_shape=[image_size,image_size,image_channels],
            classes = args.num_classes,
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # optimizer =  tfa.optimizers.AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))
    
    # tb_callback = tf.keras.callbacks.TensorBoard(os.path.join('logs', 'adamw'),
    #                                              profile_batch=0)
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # wd_callback = WeightDecayScheduler(wd_schedule)


    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy'])
    model.summary()
    # Traning
    
    model.fit(train_ds_cmu,
              epochs=epoch,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)

