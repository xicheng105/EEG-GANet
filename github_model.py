import scipy.signal as signal
import keras.backend
import tensorflow as tf
import numpy as np
import copy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D, MaxPooling1D, Conv1D,
                                     Conv2D, SeparableConv2D, DepthwiseConv2D, Layer, BatchNormalization, Reshape,
                                     Flatten, Add, Concatenate, concatenate, Lambda, Input, Permute,
                                     GlobalAveragePooling2D, ZeroPadding1D, SeparableConv1D)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K

from github_Utils import cecotti_normal, scaled_tanh, streg


# %% EEG_DBNet_V2
class EEG_DBNet_V2:
    def __init__(self, NumFilter=8, SamplingFrequency=256, NumChannels=8, FilterScaler=2, NumClasses=2,
                 DropoutRate=0.5):
        """
        :param SamplingFrequency: Int
            Sampling rate.
        :param NumChannels: Int
            Number of input channels.
        :param FilterScaler: Int
            Scaling Filters.
        :param NumClasses: Int
            Number of classifications.
        :param NumFilter: Int
            Number of filters in the first convolution layer.
        :param DropoutRate: Float
            Dropout rate.
        """

        self.NumChannels = NumChannels
        self.NumClasses = NumClasses
        self.NumSamples = SamplingFrequency
        self.SamplingFrequency = SamplingFrequency
        self.kernel_size = SamplingFrequency // 2
        self.NumFilter = NumFilter
        self.FilterScaler = FilterScaler
        self.DropoutRate = DropoutRate

    def pooling_block(self, input_layer, conv_dimension='temporal', scale_factor=32):
        if conv_dimension == 'temporal':
            output_layer = AveragePooling2D(
                pool_size=(1, int(tf.math.ceil(self.SamplingFrequency / scale_factor))),
                padding='same'
            )(input_layer)
        elif conv_dimension == 'spectral':
            output_layer = MaxPooling2D(
                pool_size=(1, int(tf.math.ceil(self.SamplingFrequency / scale_factor))),
                padding='same'
            )(input_layer)
        else:
            raise ValueError("Conv_dimension type must be 'temporal' or 'spectral'.")

        return output_layer

    def dilate_convolution_block(self, input_layer, conv_dimension='temporal', scale_factor=64):
        if conv_dimension == 'temporal':
            middle_layer = input_layer
        else:
            middle_layer = Permute((1, 3, 2))(input_layer)
        for _ in range(2):
            middle_layer = SeparableConv2D(
                filters=middle_layer.shape[-1] // 2,
                kernel_size=(1, 5),
                padding='same',
                dilation_rate=(1, 2),
                use_bias=False
            )(middle_layer)
        middle_layer = BatchNormalization()(middle_layer)
        middle_layer = Activation('elu')(middle_layer)
        output_layer = self.pooling_block(
            middle_layer,
            conv_dimension=conv_dimension,
            scale_factor=scale_factor
        )

        return output_layer

    def branch_convolution_block(self, input_layer, conv_dimension='temporal'):
        convolution_block_1 = Conv2D(
            filters=self.NumFilter,
            kernel_size=(1, self.kernel_size),
            use_bias=False,
            padding='same'
        )(input_layer)
        convolution_block_1 = BatchNormalization()(convolution_block_1)

        convolution_block_2 = DepthwiseConv2D(
            kernel_size=(self.NumChannels, 1),
            depth_multiplier=self.FilterScaler,
            use_bias=False,
            depthwise_constraint=max_norm(1.)
        )(convolution_block_1)
        convolution_block_2 = BatchNormalization()(convolution_block_2)
        convolution_block_2 = Activation('elu')(convolution_block_2)
        convolution_block_2 = self.pooling_block(
            convolution_block_2,
            conv_dimension=conv_dimension,
            scale_factor=64
        )
        convolution_block_2 = Dropout(rate=self.DropoutRate)(convolution_block_2)

        convolution_block_3 = SeparableConv2D(
            filters=self.NumFilter * self.FilterScaler,
            kernel_size=(1, self.kernel_size // 4),
            padding='same',
            use_bias=False
        )(convolution_block_2)
        convolution_block_3 = BatchNormalization()(convolution_block_3)
        convolution_block_3 = Activation('elu')(convolution_block_3)
        convolution_block_3 = self.pooling_block(
            convolution_block_3,
            conv_dimension=conv_dimension,
            scale_factor=64
        )
        convolution_block_3 = Dropout(rate=self.DropoutRate)(convolution_block_3)

        convolution_block_4 = self.dilate_convolution_block(
            convolution_block_3,
            conv_dimension=conv_dimension,
            scale_factor=64
        )
        output_layer = Dropout(rate=self.DropoutRate)(convolution_block_4)

        return output_layer

    def build_model(self):
        inputs = Input(shape=(self.NumChannels, self.NumSamples, 1))

        convolution_block_1 = self.branch_convolution_block(inputs, conv_dimension='temporal')
        convolution_block_2 = self.branch_convolution_block(inputs, conv_dimension='spectral')

        flatten_block_1 = Flatten()(convolution_block_1)
        flatten_block_2 = Flatten()(convolution_block_2)

        concatenate_block = Concatenate()([flatten_block_1, flatten_block_2])

        dense_block = Dense(self.NumClasses, kernel_constraint=max_norm(0.25))(concatenate_block)
        dense_block = Activation('sigmoid')(dense_block)

        model = Model(inputs=inputs, outputs=dense_block)

        return model
