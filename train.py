from model.convNeXt_2020s import convnext
from model.resnet50_resnet50Xt import ResNeXt, ResNet
from data import  load_dataset_original, load_dataset_cifar10
import tensorflow as tf
from argparse import ArgumentParser
# conda deactivate (trong TH đang ở môi trg khác dùng lệnh này để thoát khỏi enviroment đó r mới cài thư viện dưới nhé)
# pip install tensorflow-addons
import tensorflow_addons as tfa
import os
from optimizer_adamW import WeightDecayScheduler, lr_schedule, wd_schedule
from tensorflow.python.data import Dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import Callback


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Type of ConvNeXt model, valid option: resnet50, resnext')
    parser.add_argument('--optimizer', default="AdamW", type=str,
                        help='Type of optimizer, valid option: AdamW')
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
    optimizer = args.optimizer
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
        train_datagen,val_datagen =  load_dataset_original(args.train_folder,args.valid_folder,args.image_size,args.batch_size)
            #Load train set
        

    else:
        print("Data folder is not set. Use CIFAR 10 dataset")
        train_ds_cmu, train_ds_simple, val_ds = load_dataset_cifar10(args.batch_size,args.image_size)


    if args.model == 'resnet50':
        model = ResNet(image_size, image_size, image_channels, num_filters, problem_type=problem_type, output_nums=args.num_classes, pooling='avg', dropout_rate=False).ResNet50()
    elif args.model == 'resnext':
        model = ResNeXt(image_size, image_size, image_channels, num_filters,cardinality=32, problem_type=problem_type, output_nums=args.num_classes, pooling='avg', dropout_rate=False).ResNetXt50()
    else:
        model = convnext(
            input_shape=[image_size,image_size,image_channels],
            classes = args.num_classes,
        )

    model.summary()
    # Traning
    if optimizer == "AdamW":
        optimizer =  tfa.optimizers.AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))
        
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        
        wd_callback = WeightDecayScheduler(wd_schedule)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.fit(train_ds_cmu, validation_data=val_ds, epochs=40,
                                      callbacks=[lr_callback, wd_callback])
        #Save model
        model.save(args.model_folder)
    else: 
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
        checkpoint = ModelCheckpoint(filepath=args.model_folder + 'model.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1)
        callbacks = [learning_rate_reduction, checkpoint] 
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.fit(train_ds_cmu,
                    epochs=epoch,
                    validation_data=val_ds)

    
    
    
    
    

    

