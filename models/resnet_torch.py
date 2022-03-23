# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 14:01
# @Author  : Zeqi@@
# @FileName: resnet_torch.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.l2(
        l2_weight_decay) if use_l2_regularizer else None


use_l2_regularizer = False
l2_weight_decay = 5e-4


class ResNet():
    def __init__(self, layers_dims, nb_classes, use_bn=False, use_bias=False):
        super(ResNet, self).__init__()
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.layers_dims = layers_dims
        self.nb_classes = nb_classes

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
        x = Conv2D(filter_num, (3, 3), strides=stride, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                   bias_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                   use_bias=self.use_bias, name=conv_name_1)(inputs)
        if self.use_bn:
            x = BatchNormalization(beta_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                                   gamma_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   name=conv_name_1 + '_bn')(x)
        x = ReLU(name=conv_name_1 + '_relu')(x)

        # Part 2
        x = Conv2D(filter_num, (3, 3), strides=1, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                   bias_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                   use_bias=self.use_bias, name=conv_name_2)(x)
        if self.use_bn:
            x = BatchNormalization(beta_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                                   gamma_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   name=conv_name_2 + '_bn')(x)

        #  skip
        if stride != 1:
            residual = Conv2D(filter_num, (1, 1), strides=stride, kernel_initializer='he_normal',
                              kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                              bias_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                              use_bias=self.use_bias, name=skip_connection)(inputs)
        else:
            residual = inputs

        # Add
        x = Add(name='block_' + block_name + '_residual_add')([x, residual])
        out = ReLU(name='block_' + block_name + '_residual_add_relu')(x)

        return out

    def ConvBn_Block(self, inputs, num_filters, kernel_size, strides, block_name):
        conv_name = 'block_convbn' + block_name + '_conv'
        x = Conv2D(num_filters, kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                   bias_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                   name=conv_name)(inputs)
        x = BatchNormalization(beta_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                               gamma_regularizer=_gen_l2_regularizer(use_l2_regularizer, l2_weight_decay),
                               momentum=0.1,
                               epsilon=1e-5,
                               name=conv_name + '_bn')(x)
        out = ReLU(name=conv_name + '_relu')(x)
        return out

    def __call__(self, inputs):
        # Initial
        #  32, 32, 3 -> 32, 32, 64
        x = self.ConvBn_Block(inputs, num_filters=64, kernel_size=(3, 3), strides=(1, 1), block_name='0')

        # Basic blocks
        #  32, 32, 64 -> 32, 32, 64
        x = self.build_basic_block(x, filter_num=64, blocks=self.layers_dims[0], stride=1, module_name='module_0')
        # x = Dropout(0.2)(x)

        #  32, 32, 64 -> 16, 16, 128
        x = self.build_basic_block(x, filter_num=128, blocks=self.layers_dims[1], stride=2, module_name='module_1')
        # x = Dropout(0.2)(x)

        #  16, 16, 128 -> 8, 8, 256
        x = self.build_basic_block(x, filter_num=256, blocks=self.layers_dims[2], stride=2, module_name='module_2')
        # x = Dropout(0.4)(x)

        #  8, 8, 256 -> 4, 4, 512
        x = self.build_basic_block(x, filter_num=512, blocks=self.layers_dims[3], stride=2, module_name='module_3')
        # x = Dropout(0.5)(x)

        # Top
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='ResNet')

        return model

    # def __call__(self, inputs):
    #
    #     # Initial
    #     # 160, 480, 3 -> 80, 240, 3
    #     x = self.ConvBn_Block(inputs, num_filters=64, kernel_size=(7, 7), strides=(2, 2), block_name='0')
    #     # 80, 240, 64 -> 40, 120, 64
    #     x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    #
    #     # Basic blocks
    #     # 40, 120, 64 -> 40, 120, 64
    #     x = self.build_basic_block(x, filter_num=64,  blocks=self.layers_dims[0], stride=1, module_name='module_0')
    #     x = Dropout(0.2)(x)
    #
    #     # 40, 120, 64 -> 20, 60, 128
    #     x = self.build_basic_block(x, filter_num=128, blocks=self.layers_dims[1], stride=2, module_name='module_1')
    #     x = Dropout(0.2)(x)
    #
    #     # 20, 60, 128 -> 10, 30, 256
    #     x = self.build_basic_block(x, filter_num=256, blocks=self.layers_dims[2], stride=2, module_name='module_2')
    #     x = Dropout(0.4)(x)
    #
    #     # 10, 30, 256 -> 5, 15, 512
    #     x = self.build_basic_block(x, filter_num=512, blocks=self.layers_dims[3], stride=2, module_name='module_3')
    #     x = Dropout(0.5)(x)
    #
    #     # Top
    #     x = GlobalAveragePooling2D()(x)
    #     outputs = Dense(self.nb_classes, activation='softmax')(x)
    #
    #     model = Model(inputs=inputs, outputs=outputs, name='ResNet18')
    #
    #     return model


def build_ResNet(NetName, nb_classes, use_bn=True, use_bias=False):
    ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                     'ResNet34': [3, 4, 6, 3]}

    inputs = Input(shape=(32, 32, 3))
    ResNet_model_c = ResNet(layers_dims=ResNet_Config[NetName], nb_classes=nb_classes, use_bn=use_bn, use_bias=use_bias)
    ResNet_model = ResNet_model_c(inputs)
    return ResNet_model


def resnet18(nb_classes=100):
    model = build_ResNet('ResNet18', nb_classes=nb_classes)
    return model


def resnet34(nb_classes=100):
    model = build_ResNet('ResNet34', nb_classes=nb_classes)
    return model


if __name__ == '__main__':
    main()

# https://code.ornl.gov/olcf-analytics/summit/distributed-deep-learning-examples/-/tree/9a105bf2f7716abb0b8870cd9593ab1e99e4b7f0/examples/tensorflow/official/keras_application_models
# https://github.com/tensorflow/models/blob/master/official/vision/beta/MODEL_GARDEN.md
# chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://arxiv.org/pdf/2103.07579.pdf