import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

#（1）Khối chập đầu tiên mà hình ảnh đầu vào đi qua
def pre_Conv(inputs, out_channel):

    # 4 * 4 tích chập + chuẩn hóa
    x = layers.Conv2D(filters=out_channel,  # Số kênh của bản đồ tính năng đầu ra
                      kernel_size=(4,4),
                      strides=4,  # lấy mẫu xuống
                      padding='same')(inputs)

    x = layers.LayerNormalization()(x)

    return x

#（2）ConvNeXt Block
def block(inputs, dropout_rate=0.2, layer_scale_init_value=1e-6):
    '''
    layer_scale_init_value Giá trị khởi tạo của gama chia tỷ lệ
    '''
    # Nhận số kênh của bản đồ tính năng đầu vào
    dim = inputs.shape[-1]

    # cạnh dư
    residual = inputs

    # 7 * 7 tích chập theo chiều sâu
    x = layers.DepthwiseConv2D(kernel_size=(7,7), strides=1, padding='same')(inputs)
    # tiêu chuẩn hóa
    x = layers.LayerNormalization()(x)
    # Tích chập chuẩn 1 * 1 tăng gấp 4 lần số kênh
    x = layers.Conv2D(filters=dim*4, kernel_size=(1,1), strides=1, padding='same')(x)
    # Chức năng kích hoạt GELU
    x = layers.Activation('gelu')(x)
    # Số kênh thả chập chuẩn 1 * 1
    x = layers.Conv2D(filters=dim, kernel_size=(1,1), strides=1, padding='same')(x)
    
    # Tạo gama vectơ có thể học được, hàm này được sử dụng để thêm các biến trọng số vào một lớp, lớp khởi tạo các lớp.
    gama = layers.Layer().add_weight(shape=[dim],  # Số lượng vectơ giống như số kênh bản đồ tính năng đầu ra
                                   initializer=tf.initializers.Constant(layer_scale_init_value),  # 权重初始化
                                   dtype=tf.float32,  # chỉ định kiểu dữ liệu
                                   trainable=True)  # Các thông số có thể huấn luyện, trọng lượng có thể được điều chỉnh bằng cách nhân giống ngược

    # tỷ lệ lớp chia tỷ lệ từng dữ liệu kênh của bản đồ đối tượng và tỷ lệ chia tỷ lệ là gama
    x = x * gama  # [56,56,96]*[96]==>[56,56,96]

    # Lớp bỏ học giết chết các tế bào thần kinh một cách ngẫu nhiên
    x = layers.Dropout(rate=dropout_rate)(x)

    # Phần dư kết nối đầu vào và đầu ra
    x = layers.add([x, residual])
    
    return x

#（3）lớp downampling
def downsampling(inputs, out_channel):

    x = layers.LayerNormalization()(inputs)
    
    x = layers.Conv2D(filters=out_channel,  # Số kênh đầu ra
                      kernel_size=(2,2),
                      strides=2,  # lấy mẫu xuống
                      padding='same')(x)
    
    return x

#（4）Khối chuyển đổi, một lớp lấy mẫu xuống + nhiều lớp tích chập khối
def stage(x, num, out_channel, downsampe=True):
    '''
    num: Số lần lặp lại khối; 
    out_channel: đại diện cho số kênh đầu ra của lớp lấy mẫu xuống 
    downample: xác định xem có thực thi lớp lấy mẫu xuống hay không
    '''
    if downsampe is True:
        x = downsampling(x, out_channel)

    # Lặp lại số lần khối, mỗi lần số lượng kênh đầu ra giống nhau
    for _ in range(num):
        x = block(x)

    return x

#（5）mạng đường trục
def convnext(input_shape, classes):  # Hình dạng hình ảnh đầu vào và số loại phân loại

    # Xây dựng lớp đầu vào
    inputs = keras.Input(shape=input_shape)

    # [224,224,3]==>[56,56,96]
    x = pre_Conv(inputs, out_channel=96)
    # [56,56,96]==>[56,56,96]
    x = stage(x, num=3, out_channel=96, downsampe=False)
    # [56,56,96]==>[28,28,192]
    x = stage(x, num=3, out_channel=192, downsampe=True)
    # [28,28,192]==>[14,14,384]
    x = stage(x, num=9, out_channel=384, downsampe=True)
    # [14,14,384]==>[7,7,768]
    x = stage(x, num=3, out_channel=768, downsampe=True)

    # [7,7,768]==>[None,768]
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization()(x)

    # [None,768]==>[None,classes]
    logits = layers.Dense(classes)(x)  # softmax

    # Xây dựng mạng
    model = Model(inputs, logits)

    return model


#（6）Summary model

if __name__ == '__main__':

    # Xây dựng mạng, truyền hình ảnh đầu vào và số lượng phân loại của đầu ra cuối cùng
    model = convnext(input_shape=[224,224,3], classes=1000)

    model.summary()  # Xem cấu trúc mạng