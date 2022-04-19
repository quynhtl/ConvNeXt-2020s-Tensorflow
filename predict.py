import os
from argparse import ArgumentParser
from model.convNeXt_2020s import convnext
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    home_dir = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--test-file-path", default='{}/data/test'.format(home_dir), type=str, required=True)
    parser.add_argument("--model-folder", default='{}/model/'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    # FIXME
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ConvNeXt 2020s-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ConvNeXt 2020s model with hyper-params:') 
    print('===========================')

    # FIXME
    # Do Training
    model = tf.keras.models.load_model(args.model_folder)

    # Load test images from folder
    image = tf.keras.preprocessing.image.load_img(args.test_file_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    x = tf.image.resize(
        input_arr, [args.image_size, args.image_size]
    )
    x = x / 255
    predictions = convnext.predict(x)   
    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(predictions))
    print('This image belongs to class: {}'.format(np.argmax(predictions), axis=1))


