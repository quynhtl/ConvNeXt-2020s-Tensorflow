from model.convNeXt_2020s import convnext
from model.resnet import ResNet
from model.resnet50Xt import ResNext50 
from data import  load_dataset_original, load_dataset_cifar10
import tensorflow as tf
from argparse import ArgumentParser
# conda deactivate (trong TH đang ở môi trg khác dùng lệnh này để thoát khỏi enviroment đó r mới cài thư viện dưới nhé)
# pip install tensorflow-addons
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import Callback


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Arguments users used when running command lines
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Type of ConvNeXt model, valid option: convnext resnetxt')
    parser.add_argument('--num-classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-4,
                        type=float, help='Weight decay')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epoch')
    parser.add_argument('--image-size', default=224,
                        type=int, help='Size of input image')
    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')
    parser.add_argument('--train-folder', default='', type=str,
                        help='Where training data is located')
    parser.add_argument('--valid-folder', default='', type=str,
                        help='Where validation data is located')
    parser.add_argument('--model-folder', default='.output/',
                        type=str, help='Folder to save trained model')

    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to ConvNext-2020s paper Implementation-------------------')
    print('---------------------------------------------------------------------')
    print('Training ConvNext2020s model with hyper-params:') 
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Data loader
    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds, val_ds =  load_dataset_original(args.train_folder,args.valid_folder,args.image_size,args.batch_size)
    else:
        print("Data folder is not set. Use CIFAR 10 dataset")
        train_ds, val_ds = load_dataset_cifar10(args.image_size,args.batch_size)

    #Build model
    if args.model == 'resnet50':
        model = ResNet([3, 4, 6, 3], name='ResNet50')
        model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))
    elif args.model == 'resnetxt':
        model = ResNext50()
        model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))
    else:
        model = convnext(
            input_shape=[args.image_size, args.image_size, args.image_channels],
            classes = args.num_classes,
        )

    model.summary()

    # Traning

    optimizer = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                        patience=5, 
                                        verbose=1, 
                                        factor=0.5, 
                                        min_lr=0.00001)
    checkpoint = ModelCheckpoint(filepath=args.model_folder + 'model.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1)
    callbacks = [learning_rate_reduction, checkpoint] 
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                        metrics=['accuracy'])
    model.fit(train_ds,
                epochs=args.epochs,
                callbacks=callbacks,
                validation_data=val_ds,
                verbose=1)

    
    
    
    
    

    

