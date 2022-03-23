# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 11:27
# @Author  : Zeqi@@
# @FileName: resnet.py
# @Software: PyCharm

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras import regularizers
# from tensorflow.keras.activations import swish
# from tensorflow.keras.activations import relu
# from tensorflow.keras.activations import elu
# from tensorflow.nn import leaky_relu



def activation_func(x, name, type):
    if type == 'relu':
        out = ReLU(name=name+'_relu')(x)
    elif type == 'leaky_relu':
        '''
            f(x) = alpha * x if x < 0
            f(x) = x if x >= 0
        '''
        out = LeakyReLU(name=name+'_leaky_relu')(x)
    elif type == 'elu':
        '''
            f(x) =  alpha * (exp(x) - 1.) for x < 0
            f(x) = x for x >= 0
        '''
        out = ELU(name=name+'_elu')(x)
    # elif type == 'swish':
    #     '''
    #     x*sigmoid(x)
    #     '''
    #     out = swish(x, name=name+'_swish')
    return out

batch_norm_decay = 0.9
batch_norm_epsilon = 1e-5


class ResNet():
    def __init__(self, layers_dims, nb_classes,
                 use_bn=True, use_bias=False, **kwargs):
        super(ResNet, self).__init__()
        self.use_bn = use_bn
        self.use_bias = use_bias
        # self.use_avg = use_avg
        self.layers_dims = layers_dims
        self.nb_classes = nb_classes
        self.l2_re = kwargs.get('l2_regularization', 1e-3)
        self.type = kwargs.get('activation_type', 'relu')

    def build_basic_block(self, inputs, filter_num, blocks, stride, module_name):
        # The first block stride of each layer may be non-1
        x = self.Basic_Block(inputs, filter_num, stride, block_name='{}_{}'.format(module_name, 0))

        for i in range(1, blocks):
            x = self.Basic_Block(x, filter_num, stride=1, block_name='{}_{}'.format(module_name, i))

        return x

    def Basic_Block(self, inputs, filter_num, stride=1, block_name=None):
        conv_name_1 = 'block_' + block_name + '_conv_1'
        conv_name_2 = 'block_' + block_name + '_conv_2'
        skip_connection = 'block_' + block_name + '_skip_connection'

        # Part 1
        b_in = inputs
        x = Conv2D(filter_num, (3, 3), strides=stride, padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(self.l2_re),
                   use_bias=self.use_bias, name=conv_name_1)(b_in)
        if self.use_bn:
            x = BatchNormalization(momentum=batch_norm_decay,
                                   epsilon=batch_norm_epsilon,
                                   name=conv_name_1 + '_bn')(x)
        # x = ReLU(name=conv_name_1 + '_relu')(x)
        x = activation_func(x, conv_name_1, self.type)

        # Part 2
        x = Conv2D(filter_num, (3, 3), strides=1,
                   kernel_initializer='he_normal',
                   padding='same', kernel_regularizer=regularizers.l2(self.l2_re),
                   use_bias=self.use_bias, name=conv_name_2)(x)
        if self.use_bn:
            x = BatchNormalization(momentum=batch_norm_decay,
                                   epsilon=batch_norm_epsilon,
                                   name=conv_name_2 + '_bn')(x)

        # skip
        if stride != 1 or filter_num==128:
            residual = Conv2D(filter_num, (1, 1), strides=stride,
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(self.l2_re),
                              use_bias=self.use_bias, name=skip_connection)(b_in)
            if self.use_bn:
                residual = BatchNormalization(momentum=batch_norm_decay,
                                              epsilon=batch_norm_epsilon,
                                              name=skip_connection + '_bn')(residual)
        else:
            residual = b_in

        # Add
        x = Add(name='block_' + block_name + '_residual_add')([x, residual])
        # out = ReLU(name='block_' + block_name + '_residual_add_relu')(x)
        out = activation_func(x, 'block_' + block_name + '_residual_add', self.type)

        return out

    def ConvBn_Block(self, x, num_filters, kernel_size, strides, block_name):
        conv_name = 'block_convbn' + block_name + '_conv'

        x = Conv2D(num_filters, kernel_size,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(self.l2_re),
                   strides=strides, padding='same', name=conv_name)(x)
        if self.use_bn:
            x = BatchNormalization(momentum=batch_norm_decay,
                                   epsilon=batch_norm_epsilon,
                                   name=conv_name + '_bn')(x)
        # out = ReLU(name=conv_name + '_relu')(x)
        out = activation_func(x, conv_name, self.type)
        return out

    def __call__(self, inputs, *args, **kwargs):

        # Initial
        # 32, 32, 3 -> 32, 32, 64
        x = self.ConvBn_Block(inputs, num_filters=64, kernel_size=(7, 7), strides=(1, 1), block_name='0') # change strides

        # if self.use_avg:
        #     x = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(x)
        # else:
        #     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Basic blocks
        # 32, 32, 64 -> 32, 32, 64
        x = self.build_basic_block(x, filter_num=64, blocks=self.layers_dims[0], stride=1, module_name='module_0')
        # 32, 32, 64 -> 32, 32, 128
        x = self.build_basic_block(x, filter_num=128, blocks=self.layers_dims[1], stride=1, module_name='module_1')
        # 32, 32, 128 -> 16, 16, 256
        x = self.build_basic_block(x, filter_num=256, blocks=self.layers_dims[2], stride=2, module_name='module_2')
        # 16, 16, 256 -> 8, 8, 512
        x = self.build_basic_block(x, filter_num=512, blocks=self.layers_dims[3], stride=2, module_name='module_3')

        # Top
        # 8, 8, 512 -> 1, 1, 512
        x = GlobalAveragePooling2D()(x)
        out = Dense(self.nb_classes, name='prediction')(x)
        out = Softmax()(out)
        model = Model(inputs=inputs, outputs=out, name='ResNet')
        return model

def build_ResNet(NetName, nb_classes=10,
                 input_shape=(32, 32, 3),
                 use_bn=True, use_bias=False):

    ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                     'ResNet34': [3, 4, 6, 3]}

    inputs = Input(shape=input_shape)
    build_model = ResNet(layers_dims=ResNet_Config[NetName], nb_classes=nb_classes,
                        use_bn=use_bn, use_bias=use_bias)
    ResNet_model = build_model(inputs)
    return ResNet_model


def resnet18(NetName, nb_classes=10,
             input_shape=(32, 32, 3),
             use_bn=True, use_bias=False):
    model = build_ResNet(NetName, nb_classes, input_shape,
                         use_bn, use_bias)

    return model


