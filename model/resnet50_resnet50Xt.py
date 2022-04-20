import tensorflow as tf


def Conv_2D_Block(x, model_width, kernel, strides):
    # 2D Convolutional Block với BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def conv_block(inputs, num_filters):
    # Xây dựng khối conv không có Pooling
    conv = Conv_2D_Block(inputs, num_filters, (3, 3), (2, 2))
    conv = Conv_2D_Block(conv, num_filters, (3, 3), (1, 1))

    return conv

def stem_bottleneck(inputs, num_filters):
    # Xây dựng Stem Convolution Group
    conv = Conv_2D_Block(inputs, num_filters, (7, 7), (2, 2))
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(conv)

    return pool


def grouped_conv_block(inputs, num_filters, kernel_size, strides, cardinality):
    # thêm những nhóm convolution block
    group_list = []
    grouped_channels = int(num_filters / cardinality)

    if cardinality == 1:
        # Khi mà cardinality = 1, chỉ có standard convolution
        x = Conv_2D_Block(inputs, num_filters, 1, strides=strides)
        x = Conv_2D_Block(x, grouped_channels, kernel_size, strides)

        return x

    for c in range(cardinality):
        x = tf.keras.layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(inputs)
        x = Conv_2D_Block(x, num_filters, 1, strides=strides)
        x = Conv_2D_Block(x, grouped_channels, kernel_size, strides)

        group_list.append(x)

    group_merge = tf.keras.layers.concatenate(group_list, axis=-1)
    x = tf.keras.layers.BatchNormalization()(group_merge)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def residual_block_bottleneck(inputs, num_filters):
    # Xây dựng một khối dư của Conv (Residual Block of Convolutions)
    shortcut = Conv_2D_Block(inputs, num_filters * 4, 1, 1)
    x = Conv_2D_Block(inputs, num_filters, (1, 1), (1, 1))
    x = Conv_2D_Block(x, num_filters, (3,3), (1, 1))
    x = Conv_2D_Block(x, num_filters * 4, (1, 1), (1, 1))
    conv = tf.keras.layers.Add()([x, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out

def residual_block_bottleneck_Xt(inputs, num_filters, cardinality):
    # Xây dựng một khối dư của Conv (Residual Block of Convolutions)
    shortcut = Conv_2D_Block(inputs, num_filters * 2, (1, 1), (1, 1))
    x = grouped_conv_block(inputs, num_filters,(3,3),(1, 1), cardinality)
    x = Conv_2D_Block(x, num_filters * 2, (1, 1), (1, 1))
    conv = tf.keras.layers.Add()([x, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out

def residual_group_bottleneck(inputs, n_filters, n_blocks, conv=True):
    out = inputs
    for i in range(n_blocks):
        out = residual_block_bottleneck(out, n_filters)
    # Tăng gấp đôi size của filters và giảm feature maps còn 75% (strides=2, 2) để phù hợp với Residual Group tiếp theo
    if conv:
        out = conv_block(out, n_filters * 2)

    return out

def residual_group_bottleneck_Xt(inputs, num_filters, n_blocks, cardinality, conv=True):
    out = inputs
    for _ in range(n_blocks):
        out = residual_block_bottleneck_Xt(out, num_filters, cardinality)
    if conv:
        out = conv_block(out, num_filters * 2)

    return out

def model50(inputs, num_filters):
    # build model resnet50 :
    # First Residual Block Group of 64 filters
    # Second Residual Block Group of 128 filters
    # Third Residual Block Group of 256 filters
    # Fourth Residual Block Group of 512 filters
    x = residual_group_bottleneck(inputs, num_filters, 3)  
    x = residual_group_bottleneck(x, num_filters * 2, 3)   
    x = residual_group_bottleneck(x, num_filters * 4, 5)    
    out = residual_group_bottleneck(x, num_filters * 8, 2, False)  

    return out

def modelXt50(inputs, num_filters, cardinality):
    # build model resnet50Xt
    # 4 nhóm tuong tự modelresnet50
    x = residual_group_bottleneck_Xt(inputs, num_filters, 3, cardinality)  
    x = residual_group_bottleneck_Xt(x, num_filters * 2, 3, cardinality) 
    x = residual_group_bottleneck_Xt(x, num_filters * 4, 5, cardinality)  
    out = residual_group_bottleneck_Xt(x, num_filters * 8, 2, cardinality,False)  

    return out

def classifier(inputs, class_number):
    # Xây dựng nhóm phân loại 
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)

    return out


def regressor(inputs, feature_number):
    # Xây dựng nhóm hồi quy 
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)

    return out

def MLP(x,pooling,dropout_rate,output_nums,problem_type):
    outputs = []
    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    # Final Dense Outputting Layer cho outputs
    x = tf.keras.layers.Flatten(name='flatten')(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate, name='Dropout')(x)
    # Problem Types : Classification
    outputs = tf.keras.layers.Dense(output_nums, activation='linear')(x)
    if problem_type == 'Classification':
        outputs = tf.keras.layers.Dense(output_nums, activation='softmax')(x)

    return outputs

class ResNet:
    def __init__(self, length, width, num_channel, num_filters, check_cardinality= False, cardinality=None, problem_type='Regression', output_nums=1,
                 pooling='avg', dropout_rate=False):
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.check_cardinality = check_cardinality


    def ResNet50(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = model50(stem_b, self.num_filters)  
        outputs = MLP(x,self.pooling,self.dropout_rate,self.output_nums,self.problem_type)
        # Khởi tạo Model
        model = tf.keras.Model(inputs, outputs)

        return model


class ResNeXt:
    def __init__(self, length, width, num_channel, num_filters, cardinality=4, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False):
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.cardinality = cardinality
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate

    def ResNeXt50(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        stem_b = stem_bottleneck(inputs, self.num_filters)  # The Stem Convolution Group
        x = modelXt50(stem_b, self.num_filters, self.cardinality)
        outputs = MLP(x, self.pooling,self.dropout_rate,self.output_nums,self.problem_type)
        # khởi tạo Model
        model = tf.keras.Model(inputs, outputs)

        return model

if __name__ == '__main__':

    length = 224
    width = 224
    model_name = 'ResNetXt' 
    model_width = 16  
    num_channel = 1  
    problem_type = 'Classification' 
    output_nums = 100  
    Model = ResNeXt(length, width, num_channel, model_width, cardinality=8, problem_type=problem_type, output_nums=output_nums, pooling='avg', dropout_rate=False).ResNeXt50()
    #Model = ResNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='avg', dropout_rate=False).ResNet50()
    Model.summary()