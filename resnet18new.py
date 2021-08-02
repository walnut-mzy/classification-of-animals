import tensorflow as tf
from lajifenlei1 import rubeish_get,rubeish_get1
from class1 import class_get,class_get1
from tensorflow import keras,metrics
from tensorflow.keras import losses, optimizers,initializers
# ###这个模型构建错了，我看的网友给的模型参数有可多都是错误的，但是残差块的构建思路没错，就是这样构建的残差块
# class bottleneck(keras.layers.Layer):
#     def __init__(self,fitter1,fitter2,fitter3):
#         super(bottleneck, self).__init__()
#         self.conv2d1=keras.layers.Conv2D(filters=fitter1,kernel_size=1,strides=1,padding="same")
#         self.conv2d2=keras.layers.Conv2D(filters=fitter2,kernel_size=3,strides=1,padding="valid")
#         self.conv2d3=keras.layers.Conv2D(filters=fitter3,kernel_size=1,strides=1,padding="same")
#         self.conv2d4=keras.layers.Conv2D(filters=fitter3,kernel_size=1,strides=1,padding="same")
#         self.bn=keras.layers.BatchNormalization()
#         self.bn1=keras.layers.BatchNormalization()
#         self.bn2 = keras.layers.BatchNormalization()
#         self.relu1=keras.layers.Activation("relu")
#         self.relu2 = keras.layers.Activation("relu")
#         self.relu3 = keras.layers.Activation("relu")
#     def call(self, inputs, **kwargs):
#         print(inputs.shape)
#         x=self.conv2d1(inputs)
#         print("rest1",x.shape)
#
#         x=self.bn(x)
#         x=self.relu1(x)
#         x=self.conv2d2(x)
#         print("rest2", x.shape)
#         x=self.bn1(x)
#         x=self.relu2(x)
#         x=self.conv2d3(x)
#         print("rest3", x.shape)
#         x=self.bn2(x)
#         #变化维度
#         x1=self.conv2d4(inputs)
#         print(x.shape)
#         print(x1.shape)
#         #两个张量连接
#         x=tf.concat([x,x1],axis=2)
#         x = self.relu3(x)
#         print(x)
#         return x
# class bottleneck1(keras.layers.Layer):
#     def __init__(self,fitter1,fitter2,fitter3):
#         super(bottleneck1, self).__init__()
#         self.conv2d1=keras.layers.Conv2D(filters=fitter1,kernel_size=1,strides=1,padding="same")
#         self.conv2d2=keras.layers.Conv2D(filters=fitter2,kernel_size=3,strides=2,padding="valid")
#         self.conv2d3=keras.layers.Conv2D(filters=fitter3,kernel_size=1,strides=1,padding="same")
#         self.conv2d4=keras.layers.Conv2D(filters=fitter3,kernel_size=1,strides=2,padding="same")
#         self.bn=keras.layers.BatchNormalization()
#         self.bn1 = keras.layers.BatchNormalization()
#         self.bn2 = keras.layers.BatchNormalization()
#         self.relu1 = keras.layers.Activation("relu")
#         self.relu2 = keras.layers.Activation("relu")
#         self.relu3 = keras.layers.Activation("relu")
#     def call(self, inputs, **kwargs):
#         print(inputs.shape)
#         x = self.conv2d1(inputs)
#         x = self.bn(x)
#         x = self.relu1(x)
#         x = self.conv2d2(x)
#         x = self.bn1(x)
#         x = self.relu2(x)
#         x = self.conv2d3(x)
#         x = self.bn2(x)
#         # 变化维度
#         x1 = self.conv2d4(inputs)
#         print(x.shape)
#         print(x1.shape)
#         # 两个张量连接
#         x = tf.concat([x, x1], axis=3)
#         x = self.relu3(x)
#         return x
# class resent50(keras.layers.Layer):
#     def __init__(self):
#         super(resent50, self).__init__()
#         self.bn1=bottleneck(64,64,256)
#         self.bn2=bottleneck(64,64,256)
#         self.bn3=bottleneck(64,64,256)
#         self.bn2_1=bottleneck1(128,128,512)
#         self.bn2_2=bottleneck1(128,128,512)
#         self.bn2_3 = bottleneck1(128, 128, 512)
#         self.bn2_4 = bottleneck1(128, 128, 512)
#         self.bn3_1=bottleneck1(256,256,1024)
#         self.bn3_2 = bottleneck1(256, 256, 1024)
#         self.bn3_3 = bottleneck1(256, 256, 1024)
#         self.bn3_4 = bottleneck1(256, 256, 1024)
#         self.bn3_5 = bottleneck1(256, 256, 1024)
#         self.bn3_6 = bottleneck1(256, 256, 1024)
#         self.bn4_1 = bottleneck1(256, 256, 1024)
#         self.bn4_2 = bottleneck1(256, 256, 1024)
#         self.bn4_3 = bottleneck1(256, 256, 1024)
#         self.beg1=keras.layers.Conv2D(filters=64, kernel_size=2, padding="valid", strides=2)
#         self.beg2=keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
#     def call(self, inputs, **kwargs):
#         print(inputs.shape)
#         inputs=self.beg1(inputs)
#         print(inputs.shape)
#         inputs=self.beg2(inputs)
#         print(inputs.shape)
#         #第一个残差块
#         x=self.bn1(inputs)
#         x=self.bn2(x)
#         x=self.bn3(x)
#
#         #第二个残差块
#         x=self.bn2_1(x)
#         x=self.bn2_2(x)
#         x=self.bn2_3(x)
#         x=self.bn2_4(x)
#
#         #第三个残差块
#         x=self.bn3_1(x)
#         x=self.bn3_2(x)
#         x=self.bn3_3(x)
#         x=self.bn3_4(x)
#         x=self.bn3_5(x)
#         x=self.bn3_6(x)
#
#         #第四个残差块
#         x=self.bn4_1(x)
#         x=self.bn4_2(x)
#         x=self.bn4_3(x)
#
#         return x
#这是一个网友的renset18模型能用
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
#
# WEIGHT_DECAY = 0.01
#
#
# # def sampling(input_tensor,
# #              ksize=1,
# #              stride=2):
# #     data = input_tensor
# #     if stride > 1:
# #         data = slim.max_pool2d(data, ksize, stride=stride)
# #         print('sampling', 2)
# #     return data
#
#
# def conv2d_same(input_tensor,
#                 num_outputs,
#                 kernel_size,
#                 stride,
#                 is_train=True,
#                 activation_fn=tf.nn.relu,
#                 normalizer_fc=True
#                 ):
#     data = input_tensor
#     if stride is 1:
#         data = slim.conv2d(inputs=data,
#                            num_outputs=num_outputs,
#                            kernel_size=kernel_size,
#                            stride=stride,
#                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
#                            activation_fn=None,
#                            padding='SAME',
#                            )
#     else:
#         pad_total = kernel_size - 1
#         pad_begin = pad_total // 2
#         pad_end = pad_total - pad_begin
#         data = tf.pad(data, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
#         data = slim.conv2d(data,
#                            num_outputs=num_outputs,
#                            kernel_size=kernel_size,
#                            stride=stride,
#                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
#                            activation_fn=None,
#                            padding='VALID',
#                            )
#     if normalizer_fc:
#         data = tf.layers.batch_normalization(data, training=is_train)
#     if activation_fn is not None:
#         data = activation_fn(data)
#     return data
#
#
# def bottle_net(input_tensor, output_depth, is_train, stride=1):
#     data = input_tensor
#     depth = input_tensor.get_shape().as_list()[-1]
#     if depth == output_depth:
#         shortcut_tensor = input_tensor
#     else:
#         shortcut_tensor = conv2d_same(input_tensor, output_depth, 1, stride, is_train=is_train, activation_fn=None,
#                                       normalizer_fc=True)
#     data = conv2d_same(data, output_depth // 4, 1, 1, is_train=is_train)
#     data = conv2d_same(data, output_depth // 4, 3, stride, is_train=is_train)
#     data = conv2d_same(data, output_depth, 1, 1, is_train=is_train, activation_fn=None, normalizer_fc=False)
#
#     # 生成残差
#     data = data + shortcut_tensor
#     data = tf.nn.relu(data)
#     return data
#
#
# def create_block(input_tensor, output_depth, block_nums, init_stride=1, is_train=True, scope='block'):
#     with tf.variable_scope(scope):
#         data = bottle_net(input_tensor, output_depth, is_train=is_train, stride=init_stride)
#         for i in range(1, block_nums):
#             data = bottle_net(data, output_depth, is_train=is_train)
#         return data
#
#
# def ResNet(input_tensor, num_output, is_train, scope='resnet50'):
#     data = input_tensor
#     with tf.variable_scope(scope):
#         data = conv2d_same(data, 64, 7, 2, is_train=is_train, normalizer_fc=True)
#         data = slim.max_pool2d(data, 3, 2, padding='SAME', scope='pool_1')
#         # 第一个残差块组
#         data = create_block(data, 256, 3, init_stride=1, is_train=is_train, scope='block1')
#
#         # 第二个残差块组
#         data = create_block(data, 512, 4, init_stride=2, is_train=is_train, scope='block2')
#
#         # 第三个残差块组
#         data = create_block(data, 1024, 6, init_stride=2, is_train=is_train, scope='block3')
#
#         # 第四个残差块组
#         data = create_block(data, 2048, 3, init_stride=2, is_train=is_train, scope='block4')
#
#         # 接下来就是池化层和全连接层
#         data = slim.avg_pool2d(data, 7)
#         data = slim.conv2d(data, num_output, 1, activation_fn=None, scope='final_conv')
#
#         data_shape = data.get_shape().as_list()
#         nodes = data_shape[1] * data_shape[2] * data_shape[3]
#         data = tf.reshape(data, [-1, nodes])
#
#         return data
#
#
# if __name__ == '__main__':
#     x = tf.random_normal([32, 224, 224, 3])
#     data = ResNet(x, 1000, True)
#     print(data)
class Inception_stem(keras.layers.Layer):
    def __init__(self):
        super(Inception_stem, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=32,kernel_size=3,padding="valid",strides=2)
        self.conv2d2=keras.layers.Conv2D(filters=32, kernel_size=3, padding="valid", strides=1)
        self.conv2d3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", strides=1)
        self.maxpool1=keras.layers.MaxPool2D(pool_size=3,strides=2,padding="valid")
        self.conv2d4=keras.layers.Conv2D(kernel_size=3,strides=2,filters=96,padding="valid")
        self.conv2d5=keras.layers.Conv2D(kernel_size=1,filters=64,padding="same",strides=1)
        self.conv2d5_1 = keras.layers.Conv2D(kernel_size=1, filters=64, padding="same", strides=1)
        self.conv2d6=keras.layers.Conv2D(kernel_size=(7,1),filters=64,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(kernel_size=(1,7),filters=64,padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(kernel_size=3,filters=96,padding="valid",strides=1)
        self.conv2d10=keras.layers.Conv2D(kernel_size=1,filters=64,padding="same",strides=1)
        self.conv2d8_1 = keras.layers.Conv2D(kernel_size=3, filters=96, padding="valid", strides=1)
        self.conv2d9=keras.layers.Conv2D(kernel_size=3,filters=192,strides=2,padding="valid")
        self.maxpool2=keras.layers.MaxPool2D(pool_size=2,strides=2,padding="valid")
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        #inputs
        x=self.conv2d1(inputs)
        x=self.conv2d2(x)
        x=self.conv2d3(x)
        x1=self.maxpool1(x)
        x2=self.conv2d4(x)
        #Filter concat 73x73x160
        x=tf.concat([x1,x2],3)
        x1=self.conv2d5(x)
        x1=self.conv2d6(x1)
        x1=self.conv2d7(x1)
        x1=self.conv2d8(x1)
        x2=self.conv2d5_1(x)
        x2=self.conv2d8_1(x2)
        #Filter concat 71x71x192
        x=tf.concat([x1,x2],axis=3)
        # print("shape:", x.shape)
        x1=self.conv2d9(x)
        x2=self.maxpool2(x)
        #Filter concat 35x35x384
        x=tf.concat([x1,x2],axis=3)
        x=self.bn(x)
        # print(x)
        return x
class Inception_A(keras.layers.Layer):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding="same")
        self.conv2d2=keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding="same")
        self.conv2d3=keras.layers.Conv2D(filters=96,kernel_size=1,strides=1,padding="same")
        self.avpool=keras.layers.AveragePooling2D(padding="same",pool_size=2,strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=96,kernel_size=1,strides=1,padding="same")
        self.conv2d5=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.conv2d6=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.conv2d7=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x2=self.conv2d2(inputs)
        x3=self.conv2d3(inputs)
        x4=self.avpool(inputs)
        x4=self.conv2d4(x4)
        x2=self.conv2d5(x2)
        x1=self.conv2d6(x1)
        x1=self.conv2d7(x1)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_B(keras.layers.Layer):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(kernel_size=(1,7),filters=192,padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(kernel_size=(7,1),filters=224,padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(kernel_size=(7,1),filters=256,padding="same",strides=1)
        self.conv2d5 = keras.layers.Conv2D(kernel_size=(7, 1), filters=224, padding="same", strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(filters=224,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(filters=256,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d9=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.conv2d10=keras.layers.Conv2D(filters=128,kernel_size=1,padding="same",strides=1)
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1=self.conv2d5(x1)
        x1=self.conv2d4(x1)
        x2=self.conv2d6(inputs)
        x2=self.conv2d7(x2)
        x2=self.conv2d8(x2)
        x3=self.conv2d9(inputs)
        x4=self.avgpool(inputs)
        x4=self.conv2d10(x4)
        #print(x4.shape)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_C(keras.layers.Layer):
    def __init__(self):
        super(Inception_C, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(filters=448,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(filters=512,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=256,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d5=keras.layers.Conv2D(filters=256,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(filters=256,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(filters=256,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d9=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.conv2d10=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1_1=self.conv2d4(x1)
        x1_2=self.conv2d5(x1)
        x1=tf.concat([x1_1,x1_2],axis=3)
        x2=self.conv2d6(inputs)
        x2_1=self.conv2d7(x2)
        x2_2=self.conv2d8(x2)
        x2=tf.concat([x2_1,x2_2],axis=3)
        x3=self.conv2d9(inputs)
        x4=self.avgpool(inputs)
        x4=self.conv2d10(x4)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_redution_A(keras.layers.Layer):
    def __init__(self):
        super(Inception_redution_A, self).__init__()
        #这里有两三种网络结构，这里我们直接使用Inception_v4模块
        self.conv2d1=keras.layers.Conv2D(kernel_size=1,filters=192,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(kernel_size=3,filters=224,padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(kernel_size=3,filters=256,padding="valid",strides=2)
        self.conv2d4=keras.layers.Conv2D(kernel_size=3,filters=384,padding="valid",strides=2)
        self.maxpool=keras.layers.MaxPool2D(pool_size=3,strides=2,padding="valid")
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x2=self.conv2d4(inputs)
        x3=self.maxpool(inputs)
        x=tf.concat([x1,x2,x3],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_redution_B(keras.layers.Layer):
    def __init__(self):
        super(Inception_redution_B, self).__init__()
        #这里有两三种网络结构，这里我们直接使用Inception_v4模块
        self.conv2d1=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(filters=256,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(filters=320,kernel_size=(7,1),padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=320,kernel_size=3,padding="valid",strides=2)
        self.conv2d5=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=192,kernel_size=3,padding="valid",strides=2)
        self.maxpool=keras.layers.MaxPool2D(strides=2,padding="valid",pool_size=3)
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1=self.conv2d4(x1)
        x2=self.conv2d5(inputs)
        x2=self.conv2d6(x2)
        x3=self.maxpool(inputs)
        x=tf.concat([x1,x2,x3],axis=3)
        #print(x)
        return x
class Inception(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        #self.attention_dim = attention_dim
        super(Inception, self).__init__()
        self.stem=Inception_stem()
        self.Inception1=Inception_A()
        self.Inception2=Inception_A()
        self.Inception3=Inception_A()
        self.Inception4=Inception_A()
        self.reduction_A=Inception_redution_A()
        self.Inception_b1=Inception_B()
        self.Inception_b2=Inception_B()
        self.Inception_b3=Inception_B()
        self.Inception_b4=Inception_B()
        self.Inception_b5=Inception_B()
        self.Inception_b6=Inception_B()
        self.Inception_b7=Inception_B()
        self.reduction_B=Inception_redution_B()
        self.Inception_c1=Inception_C()
        self.Inception_c2=Inception_C()
        self.Inception_c3=Inception_C()
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.droupout=keras.layers.Dropout(0.2)
        self.fl=keras.layers.Flatten()
        self.soft=keras.layers.Dense(5,activation="softmax")
        self.bn=keras.layers.BatchNormalization()
        self.bn1=keras.layers.BatchNormalization()
        self.bn2=keras.layers.BatchNormalization()
        self.bn3=keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()
        #self.reshape=keras.layers.Reshape([])
    def call(self, inputs, **kwargs):
        x=self.stem(inputs)
        #print("stem:",x.shape)
        x=self.Inception1(x)
        #print("Inception1:",x.shape)
        x=self.Inception2(x)
        x=self.Inception3(x)
        x=self.Inception4(x)
        #print("inception4:",x.shape)
        x=self.reduction_A(x)

        x=self.bn(x)
        x=self.Inception_b1(x)
        x=self.Inception_b2(x)
        x=self.Inception_b3(x)
        x=self.Inception_b4(x)
        x=self.bn1(x)
        x=self.Inception_b5(x)
        x=self.Inception_b6(x)
        x=self.Inception_b7(x)
        x=self.reduction_B(x)
        x=self.bn2(x)
        x=self.Inception_c1(x)
        x=self.Inception_c2(x)
        x=self.Inception_c3(x)
        x=self.bn3(x)
        x=self.avgpool(x)
        x=self.droupout(x)
        x=self.fl(x)
        x=self.bn4(x)
        #print(x.shape)
        x=self.soft(x)
        #x=self.reshape(x) #这个函数的作用是使其于标签属性一致
        #print(x)
        return x

    # def get_config(self):
    #     config = {
    #         'attention_dim': self.attention_dim
    #     }
    #     base_config = super(Inception, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    # model1 = tf.keras.applications.ResNet50(
    #     include_top=False,
    #     weights='imagenet',
    #     input_shape=(224, 224, 3),
    # )
    # model = tf.keras.Sequential([
    #    model1,
    #     tf.keras.layers.BatchNormalization(),
    #    tf.keras.layers.AveragePooling2D(padding="valid",pool_size=5),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(5,activation="softmax"),
    # ])
    model=tf.keras.Sequential([
        Inception(),
    ])
    #model=tf.saved_model.load("model-savedmodel")
    loss_object = losses.categorical_crossentropy
    acc_meter = metrics.CategoricalAccuracy()
    optimizer = optimizers.Adam(lr=0.001)
    #
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=['accuracy']
    )
    model.build(input_shape=(None, 299, 299, 3))
    model.summary()
    # dataset=rubeish_get()
    dataset1=class_get()
    dataset=class_get1()

    # for epoch in range(15):
    #     for x, y in dataset:
    #         # print(x.shape)
    #         # print(y.shape)
    #         with tf.GradientTape() as tape:
    #             predictions = model(x)
    #             acc_meter1.update_state(y_true=y, y_pred=predictions)
    #             loss = loss_object(y, predictions)
    #             # loss1(y_true=y, y_pred=predictions)
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         # 打印准确率
    #         print("Test Accuracy:%f" % acc_meter.result())
    #         print("epoch{} train_loss is {};train_accuracy is {};test_accuracy is {}".format(epoch + 1,
    #                                                                                          loss[0],
    #                                                                                          acc_meter1.result(),
    #                                                                                          acc_meter.result(),
    #                                                                                          ))
    #     for x1, y1 in dataset1:  # 遍历测试集
    #         pred = model(x1)  # 前向计算
    #         acc_meter.update_state(y_true=y1, y_pred=pred)  # 更新准确率统计
    for epoch in range(1,15):
        model.fit(dataset,batch_size=100)
        if epoch==5 or epoch==10:

            tf.saved_model.save(model, 'model-savedmodel')
            print('saving savedmodel.')
            for x1, y1 in dataset1:  # 遍历测试集
                pred = model(x1)  # 前向计算
                acc_meter.update_state(y_true=y1, y_pred=pred)  # 更新准确率统计
            print("测试集正确率为：", acc_meter.result())
            acc_meter.reset_states()
