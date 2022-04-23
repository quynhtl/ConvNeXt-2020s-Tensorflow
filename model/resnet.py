import tensorflow as tf

from keras import Input, Model, Sequential
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, Dense, Flatten, add

"""
input
conv ~ 7 x 7 kernel_size, filters = 64, stride = 2
maxpool ~ pool size = 3, strides = 2
[1 x 1, 64]
[3 x 3, 64] x 3           [CONV #2]
[1 x 1, 256]
[1 x 1, 128]
[3 x 3, 128] x 4          [CONV #3]
[1 x 1, 512]
[1 x 1, 256]
[3 x 3, 256] x 6          [CONV #4]
[1 x 1, 1024]
[1 x 1, 512]
[3 x 3, 512] x 3          [CONV #5]
[1 x 1, 2048]

Downsampling-Việc lấy mẫu ở CONV # 3, # 4 và # 5 được thực hiện bằng stride-2 convolutions 
trong lớp 3 x 3 của Convolution block đầu tiên trong mỗi giai đoạn. 
Lớp đầu tiên là khối Convolution cho mọi giai đoạn theo sau là identity block
"""

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, use_conv=False):
        super(Block, self).__init__()
        self.filters = filters
        self.use_conv = use_conv
        self.strides = strides

        self.conv1 = Conv2D(filters, kernel_size=1)
        self.bn1 = BatchNormalization()

        if use_conv:
            self.conv2 = Conv2D(filters, kernel_size=3, strides=strides, padding='same')
        else:
            self.conv2 = Conv2D(filters, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(4*filters, kernel_size=1)
        self.bn3 = BatchNormalization()
        self.relu = ReLU()

        self.conv_block = Conv2D(filters=4*self.filters, kernel_size=1, strides=self.strides)
        self.conv_bn = BatchNormalization()

    def call(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.use_conv:
            identity = self.conv_block(identity)
            identity = self.conv_bn(identity)
        x = add([x, identity])
        x = self.relu(x)
        return x

class RESLayer(tf.keras.Model):
    def __init__(self, block_multiples, first_block=False):
        super(RESLayer, self).__init__()
        self.block_multiples = block_multiples
        self.residual_layers = []
        f = 64
        for stage, block_multiple in enumerate(block_multiples):
            s = 1 if stage == 0 else 2
            self.residual_layers.append(Block(f, s, use_conv=True))
            for r in range(block_multiple - 1):
                self.residual_layers.append(Block(f))
            f *= 2

    def call(self, x):
        for layer in self.residual_layers.layers:
            x = layer(x)
        return x

def ResNet(type_of_net, name=None, n_classes=2):
    return Sequential([
        Conv2D(filters=64, kernel_size=7, strides=2, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPool2D(pool_size=3, strides=2, padding='same'),
        RESLayer(type_of_net),
        GlobalAvgPool2D(),
        Flatten(),
        Dense(n_classes, activation='softmax')
    ], name=name)


# resnet50 = ResNet([3, 4, 6, 3], name='ResNet50')
# resnet50.summary()


if __name__ == '__main__':
    model = ResNet([3, 4, 6, 3], name='ResNet50')
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
    # train_ds_cmu,val_ds = load_dataset_original(train_folder,valid_folder,length,64)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # model.compile(optimizer=optimizer, loss='categorical_crossentropy',
    #            metrics=['accuracy'])
    # model.fit(train_ds_cmu,
    #             epochs=20,
    #             validation_data=val_ds)