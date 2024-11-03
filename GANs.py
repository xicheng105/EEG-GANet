import time
import numpy as np

from tensorflow.keras.layers import Input, Dense, Reshape, Permute, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras.layers import Activation, Dropout, LeakyReLU, GroupNormalization
from tensorflow.keras.layers import UpSampling1D, Concatenate, Flatten, MaxPooling2D
from tensorflow.keras.layers import Conv1DTranspose, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

import tensorflow as tf


# %% Generators.
class GAN_generator_1:
    def __init__(self, SamplingFrequency=256, NumChannels=8, Stride=2, ActivationType='leaky_relu',
                 NormalizationType='GroupNormalization'):

        self.SamplingFrequency = SamplingFrequency
        self.NumChannels = NumChannels
        self.Stride = Stride
        self.ActivationType = ActivationType
        self.NormalizationType = NormalizationType

    def conv_1d(self, InputLayer, NumFilters, KernelSize=3, Padding='same'):
        down_sampling_layer = Conv1D(
            filters=NumFilters,
            kernel_size=KernelSize,
            strides=self.Stride,
            padding=Padding
        )(InputLayer)

        if self.NormalizationType == 'BatchNormalization':
            normalization_layer = BatchNormalization()(down_sampling_layer)
        elif self.NormalizationType == 'GroupNormalization':
            normalization_layer = GroupNormalization(groups=-1)(down_sampling_layer)
        else:
            raise Exception("'{}' normalize is not supported yet!".format(self.NormalizationType))

        if self.ActivationType == "elu":
            activate_layer = Activation("elu")(normalization_layer)
        elif self.ActivationType == "leaky_relu":
            activate_layer = LeakyReLU(alpha=0.2)(normalization_layer)
        else:
            raise Exception("'{}' activation function is not supported yet!".format(self.ActivationType))

        return activate_layer

    def de_conv_1d(self, InputLayer, ResidualBlock, NumFilters, KernelSize=3, Padding='same'):
        up_sampling_block = UpSampling1D(size=self.Stride)(InputLayer)
        convolution_layer = Conv1D(
            filters=NumFilters,
            kernel_size=KernelSize,
            strides=1,
            padding=Padding
        )(up_sampling_block)

        if self.NormalizationType == 'BatchNormalization':
            normalization_layer = BatchNormalization()(convolution_layer)
        elif self.NormalizationType == 'GroupNormalization':
            normalization_layer = GroupNormalization(groups=-1)(convolution_layer)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(self.NormalizationType))

        if self.ActivationType == "elu":
            activate_layer = Activation("elu")(normalization_layer)
        elif self.ActivationType == "leaky_relu":
            activate_layer = LeakyReLU(alpha=0.2)(normalization_layer)
        else:
            raise Exception("'{}' activation function is not supported yet!".format(self.ActivationType))
        concatenation_layer = Concatenate()([activate_layer, ResidualBlock])

        return concatenation_layer

    def build_model(self):
        inputs = Input(shape=(self.NumChannels, self.SamplingFrequency))
        input_layer = Permute((2, 1))(inputs)  # (256, 8)

        # Down sampling
        down_sampling_block_1 = Conv1D(
            filters=self.NumChannels * 2,
            kernel_size=3,
            padding='same',
            strides=self.Stride,
            activation=self.ActivationType
        )(input_layer)  # (128, 16)

        down_sampling_block_2 = self.conv_1d(
            InputLayer=down_sampling_block_1,
            NumFilters=self.NumChannels * 4
        )  # (64, 32)

        down_sampling_block_3 = self.conv_1d(
            InputLayer=down_sampling_block_2,
            NumFilters=self.NumChannels * 8
        )  # (32, 64)

        down_sampling_block_4 = self.conv_1d(
            InputLayer=down_sampling_block_3,
            NumFilters=self.NumChannels * 16
        )  # (16, 128)

        down_sampling_block_5 = self.conv_1d(
            InputLayer=down_sampling_block_4,
            NumFilters=self.NumChannels * 32
        )  # (8, 256)

        down_sampling_block_6 = self.conv_1d(
            InputLayer=down_sampling_block_5,
            NumFilters=self.NumChannels * 32
        )  # (4, 256)

        down_sampling_block_7 = self.conv_1d(
            InputLayer=down_sampling_block_6,
            NumFilters=self.NumChannels * 32,
            KernelSize=2,
            Padding='valid'
        )  # (2, 256)

        down_sampling_block_8 = self.conv_1d(
            InputLayer=down_sampling_block_7,
            NumFilters=self.NumChannels * 32,
            KernelSize=2,
            Padding='valid'
        )  # (1, 256)

        # Up sampling
        up_sampling_block_1 = self.de_conv_1d(
            InputLayer=down_sampling_block_8,
            ResidualBlock=down_sampling_block_7,
            NumFilters=down_sampling_block_7.shape[-1]
        )  # (2, 256) Con (2, 256) = (2, 512)

        up_sampling_block_2 = self.de_conv_1d(
            InputLayer=up_sampling_block_1,
            ResidualBlock=down_sampling_block_6,
            NumFilters=down_sampling_block_6.shape[-1]
        )  # (4, 256) Con (4, 256) = (4, 512)

        up_sampling_block_3 = self.de_conv_1d(
            InputLayer=up_sampling_block_2,
            ResidualBlock=down_sampling_block_5,
            NumFilters=down_sampling_block_5.shape[-1]
        )  # (8, 256) Con (8, 256) = (8, 512)

        up_sampling_block_4 = self.de_conv_1d(
            InputLayer=up_sampling_block_3,
            ResidualBlock=down_sampling_block_4,
            NumFilters=down_sampling_block_4.shape[-1]
        )  # (16, 128) Con (16, 128) = (16, 256)

        up_sampling_block_5 = self.de_conv_1d(
            InputLayer=up_sampling_block_4,
            ResidualBlock=down_sampling_block_3,
            NumFilters=down_sampling_block_3.shape[-1]
        )  # (32, 64) Con (32, 64) = (32, 128)

        up_sampling_block_6 = self.de_conv_1d(
            InputLayer=up_sampling_block_5,
            ResidualBlock=down_sampling_block_2,
            NumFilters=down_sampling_block_2.shape[-1]
        )  # (64, 32) Con (64, 32) = (64, 64)

        up_sampling_block_7 = self.de_conv_1d(
            InputLayer=up_sampling_block_6,
            ResidualBlock=down_sampling_block_1,
            NumFilters=down_sampling_block_1.shape[-1]
        )  # (128, 16) Con (128, 16) = (128, 32)

        up_sampling_block_8 = UpSampling1D(size=2)(up_sampling_block_7)  # (256, 32)

        out_put = Conv1D(
            filters=self.NumChannels,
            kernel_size=3,
            padding='same'
        )(up_sampling_block_8)  # (256, 8)

        output_layer = Permute((2, 1))(out_put)

        model = Model(inputs=inputs, outputs=output_layer)

        return model


class GAN_generator_2:
    def __init__(self, SamplingFrequency=256, NumChannels=8, ConvStride=3):
        self.NumChannels = NumChannels
        self.NumSamples = SamplingFrequency
        self.ZDim = NumChannels * SamplingFrequency // 100
        self.ConvStride = ConvStride

    def build_model(self):
        inputs = Input(shape=self.ZDim)

        dense_block = Dense(
            units=self.ZDim * self.NumChannels,
            activation='PReLU',
            kernel_initializer='glorot_uniform'
        )(inputs)
        dense_block = Reshape((self.ZDim, self.NumChannels))(dense_block)

        transpose_convolution_block_1 = Conv1DTranspose(
            filters=self.NumChannels * 2,
            kernel_size=int(self.ZDim // 2),
            strides=self.ConvStride,
            activation='PReLU',
            kernel_initializer='glorot_uniform'
        )(dense_block)

        transpose_convolution_block_2 = Conv1DTranspose(
            filters=self.NumChannels * 2,
            kernel_size=int(self.ZDim // 2),
            strides=1,
            activation='PReLU',
            kernel_initializer='glorot_uniform'
        )(transpose_convolution_block_1)

        transpose_convolution_block_3 = Conv1DTranspose(
            filters=self.NumChannels,
            kernel_size=self.NumSamples - (transpose_convolution_block_2.shape[-2] - 1) * self.ConvStride,
            strides=self.ConvStride,
            activation='PReLU',
            kernel_initializer='glorot_uniform'
        )(transpose_convolution_block_2)

        PermuteLayer = Permute((2, 1))(transpose_convolution_block_3)

        model = Model(inputs=inputs, outputs=PermuteLayer)

        return model


class GAN_generator_3:
    def __init__(self, SamplingFrequency=256, NumChannels=8, ConvStride=3):
        self.NumChannels = NumChannels
        self.NumSamples = SamplingFrequency
        self.ZDim = NumChannels * SamplingFrequency // 100
        self.ConvStride = ConvStride

    def encoder(self, input_layer):
        convolution_block_1 = Conv1D(
            filters=self.NumChannels * 2,
            kernel_size=int(self.NumSamples // 25),
            strides=2,
            data_format="channels_first",
            kernel_initializer='glorot_uniform'
        )(input_layer)
        convolution_block_1 = BatchNormalization(axis=1)(convolution_block_1)
        convolution_block_1 = MaxPooling1D(pool_size=2, data_format="channels_first")(convolution_block_1)

        flatten_block = Flatten()(convolution_block_1)

        mu = Dense(
            units=self.ZDim,
            kernel_initializer='glorot_uniform'
        )(flatten_block)
        logvar = Dense(
            units=self.ZDim,
            kernel_initializer='glorot_uniform'
        )(flatten_block)

        std = tf.exp(0.5 * logvar)
        esp = tf.random.normal(shape=tf.shape(mu), dtype=mu.dtype)
        z = esp * std + mu

        return z, mu, logvar

    def decoder(self, input_layer):
        dense_block = Dense(
            units=self.ZDim * self.NumChannels,
            kernel_initializer='glorot_uniform'
        )(input_layer)
        reshape_block = Reshape((self.ZDim, self.NumChannels))(dense_block)

        transpose_convolution_block_1 = Conv1DTranspose(
            filters=self.NumChannels * 2,
            kernel_size=int(self.ZDim // 2),
            strides=self.ConvStride,
            activation='PReLU',
            kernel_initializer='glorot_uniform'
        )(reshape_block)

        transpose_convolution_block_2 = Conv1DTranspose(
            filters=self.NumChannels * 2,
            kernel_size=int(self.ZDim // 2),
            strides=1,
            activation='PReLU',
            kernel_initializer='glorot_uniform'
        )(transpose_convolution_block_1)

        transpose_convolution_block_3 = Conv1DTranspose(
            filters=self.NumChannels,
            kernel_size=self.NumSamples - (transpose_convolution_block_2.shape[-2] - 1) * self.ConvStride,
            strides=self.ConvStride,
            activation='sigmoid',
            kernel_initializer='glorot_uniform'
        )(transpose_convolution_block_2)

        return transpose_convolution_block_3

    def build_model(self):
        inputs = Input(shape=(self.NumChannels, self.NumSamples))
        z, mu, logvar = self.encoder(inputs)
        transpose_convolution_block_3 = self.decoder(z)
        PermuteLayer = Permute((2, 1))(transpose_convolution_block_3)

        model = Model(inputs=inputs, outputs=[PermuteLayer, mu, logvar])

        return model


# %% Discriminators.
class GAN_discriminator_1:
    def __init__(self, NumFilter=8, SamplingFrequency=256, NumChannels=8, FilterScaler=2, DropoutRate=0.5,
                 IsWasserstein=False):

        self.NumChannels = NumChannels
        self.NumSamples = SamplingFrequency
        self.SamplingFrequency = SamplingFrequency
        self.kernel_size = SamplingFrequency // 2
        self.NumFilter = NumFilter
        self.FilterScaler = FilterScaler
        self.DropoutRate = DropoutRate
        self.IsWasserstein = IsWasserstein

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

        dense_block = Dense(1, kernel_constraint=max_norm(0.25))(concatenate_block)
        dense_block = Activation('sigmoid')(dense_block)

        model = Model(inputs=inputs, outputs=dense_block)

        return model


class GAN_discriminator_2:
    def __init__(self, SamplingFrequency=256, NumChannels=8, IsWasserstein=False):
        self.NumChannels = NumChannels
        self.NumSamples = SamplingFrequency
        self.IsWasserstein = IsWasserstein

    def build_model(self):
        inputs = Input(shape=(self.NumChannels, self.NumSamples))

        convolution_block = Conv1D(
            filters=self.NumChannels * 2,
            kernel_size=int(self.NumSamples // 25),
            strides=2,
            data_format="channels_first",
            activation="LeakyReLU",
            kernel_initializer='glorot_uniform'
        )(inputs)
        convolution_block = BatchNormalization(axis=1)(convolution_block)
        convolution_block = MaxPooling1D(pool_size=2, data_format="channels_first")(convolution_block)

        dense_block = Flatten()(convolution_block)

        if self.IsWasserstein:
            dense_block = Dense(
                units=1,
                kernel_initializer='glorot_uniform'
            )(dense_block)
        else:
            dense_block = Dense(
                units=1,
                activation="sigmoid",
                kernel_initializer='glorot_uniform'
            )(dense_block)

        model = Model(inputs=inputs, outputs=dense_block)

        return model


class GAN_discriminator_3:
    def __init__(self, NumFilter=8, SamplingFrequency=256, NumChannels=8, KernelSize=3, Strides=2):
        self.NumChannels = NumChannels
        self.NumSamples = SamplingFrequency
        self.SamplingFrequency = SamplingFrequency
        self.KernelSize = KernelSize
        self.NumFilter = NumFilter
        self.Stride = Strides

    def conv_1d(self, InputLayer, NumFilters, KernelSize=3, Padding='same'):
        down_sampling_layer = Conv1D(
            filters=NumFilters,
            kernel_size=KernelSize,
            strides=self.Stride,
            padding=Padding
        )(InputLayer)
        normalization_layer = BatchNormalization()(down_sampling_layer)
        activate_layer = LeakyReLU(alpha=0.2)(normalization_layer)

        return activate_layer

    def build_model(self):
        inputs = Input(shape=(self.NumChannels, self.SamplingFrequency))
        input_layer = Permute((2, 1))(inputs)  # (256, 8)

        # Down sampling
        down_sampling_block_1 = Conv1D(
            filters=self.NumChannels * 2,
            kernel_size=self.KernelSize,
            strides=self.Stride,
            padding='same',
            activation='LeakyReLU'
        )(input_layer)  # (128, 16)

        down_sampling_block_2 = self.conv_1d(
            InputLayer=down_sampling_block_1,
            NumFilters=self.NumChannels * 4,
        )  # (64, 32)

        down_sampling_block_3 = self.conv_1d(
            InputLayer=down_sampling_block_2,
            NumFilters=self.NumChannels * 8,
        )  # (32, 64)

        down_sampling_block_4 = self.conv_1d(
            InputLayer=down_sampling_block_3,
            NumFilters=self.NumChannels * 8,
        )  # (16, 64)

        down_sampling_block_5 = self.conv_1d(
            InputLayer=down_sampling_block_4,
            NumFilters=self.NumChannels * 8,
        )  # (8, 64)

        output_layer = Conv1D(
            filters=1,
            kernel_size=self.KernelSize,
            strides=self.Stride // 2,
            padding='same'
        )(down_sampling_block_5)

        model = Model(inputs=inputs, outputs=output_layer)

        return model


# %% DiscoGAN
def DiscoGAN(OriginalData, OriginalLabel, SubjectIdx, SamplingFrequency, NumChannels=8, NumAugEpochs=1000,
             BatchSize=64, AugmentRate=1, ActivationType='leaky_relu', NormalizationType='GroupNormalization'):
    real_target_data = OriginalData[OriginalLabel == 1]
    real_nontarget_data = OriginalData[OriginalLabel == 0]

    valid_rates = [i / 10 for i in range(1, 11)]
    if AugmentRate not in valid_rates:
        raise ValueError(f"AugmentRate must be one of the following values: {valid_rates}")
    augment_data = int((real_nontarget_data.shape[0] - real_target_data.shape[0]) * AugmentRate)

    StartTime = time.time()
    print("\n*************************************************************************************************")
    print("Augment {} EEG data for target of subject {}, by the 'DiscoGAN', in {} epochs.".
          format(augment_data, SubjectIdx, NumAugEpochs))
    # print("*************************************************************************************************\n")

    # Initialize generators
    generator_AB = GAN_generator_1(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels,
        ActivationType=ActivationType,
        NormalizationType=NormalizationType
    ).build_model()

    generator_BA = GAN_generator_1(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels,
        ActivationType=ActivationType,
        NormalizationType=NormalizationType
    ).build_model()

    # Initialize and compile discriminators
    discriminator_A = GAN_discriminator_1(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels
    ).build_model()
    discriminator_A.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0001)
    )

    discriminator_B = GAN_discriminator_1(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels
    ).build_model()
    discriminator_B.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0001)
    )

    # For the combined model, we will only train the generators
    discriminator_A.trainable = False
    discriminator_B.trainable = False

    # Create the combined model (generator + discriminator)
    real_nontarget_data_input = Input(shape=(NumChannels, SamplingFrequency))
    real_target_data_input = Input(shape=(NumChannels, SamplingFrequency))

    fake_target_data = generator_AB(real_nontarget_data_input)
    fake_nontarget_data = generator_BA(real_target_data_input)

    reconstructed_nontarget_data = generator_BA(fake_target_data)
    reconstructed_target_data = generator_AB(fake_nontarget_data)

    validity_A = discriminator_A(fake_nontarget_data)
    validity_B = discriminator_B(fake_target_data)

    combined_model = Model(
        inputs=[
            real_nontarget_data_input, real_target_data_input
        ],
        outputs=[
            validity_B, validity_A,
            reconstructed_nontarget_data, reconstructed_target_data
        ]
    )
    combined_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=[
            'binary_crossentropy', 'binary_crossentropy',
            'mae', 'mae'
        ],
        loss_weights=[
            1.0, 1.0,
            0.1, 0.1
        ]
    )

    for epoch in range(NumAugEpochs):
        # Prepare data for train.
        tar_idx = np.random.randint(0, real_target_data.shape[0], BatchSize)
        real_target_data_use = real_target_data[tar_idx]
        non_idx = np.random.randint(0, real_nontarget_data.shape[0], BatchSize)
        real_nontarget_data_use = real_nontarget_data[non_idx]

        # Translate data to opposite domain.
        fake_target_data_use = generator_AB(real_nontarget_data_use)
        fake_nontarget_data_use = generator_BA(real_target_data_use)

        # Train the discriminators
        discriminator_A_loss_real = discriminator_A.train_on_batch(
            real_nontarget_data_use,
            np.ones(BatchSize)
        )
        discriminator_A_loss_fake = discriminator_A.train_on_batch(
            fake_nontarget_data_use,
            np.zeros(BatchSize)
        )
        discriminator_A_loss = 0.5 * np.add(discriminator_A_loss_real, discriminator_A_loss_fake)

        discriminator_B_loss_real = discriminator_B.train_on_batch(
            real_target_data_use,
            np.ones(BatchSize)
        )
        discriminator_B_loss_fake = discriminator_B.train_on_batch(
            fake_target_data_use,
            np.zeros(BatchSize)
        )
        discriminator_B_loss = 0.5 * np.add(discriminator_B_loss_real, discriminator_B_loss_fake)

        discriminator_loss = 0.5 * np.add(discriminator_A_loss, discriminator_B_loss)

        # Train the generators
        generator_loss = combined_model.train_on_batch(
            [
                real_nontarget_data_use, real_target_data_use
            ],
            [
                np.ones(BatchSize), np.ones(BatchSize),
                real_nontarget_data_use, real_target_data_use
            ]
        )

        # if epoch % 100 == 0:
        #     print(f"{epoch} "
        #           f"[Discriminator loss: {discriminator_loss:.4f}] "
        #           f"[Generator loss: {generator_loss[0]:.4f}, "
        #           f"Validity_B loss: {generator_loss[1]:.4f}, "
        #           f"Validity_A loss: {generator_loss[2]:.4f}, "
        #           f"ReconNonTarget loss: {generator_loss[3]:.4f}, "
        #           f"ReconTarget loss: {generator_loss[4]:.4f}]")

    nontarget_idx = np.random.randint(0, real_nontarget_data.shape[0], augment_data)
    real_nontarget_data_transform = real_nontarget_data[nontarget_idx]
    augment_target_data = generator_AB.predict(real_nontarget_data_transform)

    # print(f"Training completed in {(time.time() - StartTime) / 60:.2f} minutes.\n")

    return np.array(augment_target_data), generator_AB


# %% DiscoGANX
def DiscoGANX(OriginalData, OriginalLabel, SubjectIdx, SamplingFrequency, NumChannels=8, NumAugEpochs=1000,
              BatchSize=64, AugmentRate=1, ActivationType='leaky_relu', NormalizationType='GroupNormalization'):
    real_target_data = OriginalData[OriginalLabel == 1]
    real_nontarget_data = OriginalData[OriginalLabel == 0]

    valid_rates = [i / 10 for i in range(1, 11)]
    if AugmentRate not in valid_rates:
        raise ValueError(f"AugmentRate must be one of the following values: {valid_rates}")
    augment_data = int((real_nontarget_data.shape[0] - real_target_data.shape[0]) * AugmentRate)

    StartTime = time.time()
    print("\n*************************************************************************************************")
    print("Augment {} EEG data for target of subject {}, by the 'DiscoGANX', in {} epochs.".
          format(augment_data, SubjectIdx, NumAugEpochs))
    print("*************************************************************************************************\n")

    # Initialize generators
    generator_AB = GAN_generator_1(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels,
        ActivationType=ActivationType,
        NormalizationType=NormalizationType
    ).build_model()

    generator_BA = GAN_generator_1(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels,
        ActivationType=ActivationType,
        NormalizationType=NormalizationType
    ).build_model()

    # Initialize and compile discriminators
    discriminator_A = GAN_discriminator_3(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels
    ).build_model()
    discriminator_A.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0001)
    )

    discriminator_B = GAN_discriminator_3(
        SamplingFrequency=SamplingFrequency,
        NumChannels=NumChannels
    ).build_model()
    discriminator_B.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0001)
    )

    # For the combined model, we will only train the generators
    discriminator_A.trainable = False
    discriminator_B.trainable = False

    # Create the combined model (generator + discriminator)
    real_nontarget_data_input = Input(shape=(NumChannels, SamplingFrequency))
    real_target_data_input = Input(shape=(NumChannels, SamplingFrequency))

    fake_target_data = generator_AB(real_nontarget_data_input)
    fake_nontarget_data = generator_BA(real_target_data_input)

    reconstructed_nontarget_data = generator_BA(fake_target_data)
    reconstructed_target_data = generator_AB(fake_nontarget_data)

    validity_A = discriminator_A(fake_nontarget_data)
    validity_B = discriminator_B(fake_target_data)

    combined_model = Model(
        inputs=[
            real_nontarget_data_input, real_target_data_input
        ],
        outputs=[
            validity_B, validity_A,
            reconstructed_nontarget_data, reconstructed_target_data
        ]
    )
    combined_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=[
            'binary_crossentropy', 'binary_crossentropy',
            'mae', 'mae'
        ],
        loss_weights=[
            1.0, 1.0,
            0.1, 0.1
        ]
    )

    for epoch in range(NumAugEpochs):
        # Prepare data for train.
        tar_idx = np.random.randint(0, real_target_data.shape[0], BatchSize)
        real_target_data_use = real_target_data[tar_idx]
        non_idx = np.random.randint(0, real_nontarget_data.shape[0], BatchSize)
        real_nontarget_data_use = real_nontarget_data[non_idx]

        # Translate data to opposite domain.
        fake_target_data_use = generator_AB(real_nontarget_data_use)
        fake_nontarget_data_use = generator_BA(real_target_data_use)

        # Train the discriminators
        discriminator_A_loss_real = discriminator_A.train_on_batch(
            real_nontarget_data_use,
            np.ones((BatchSize,) + (8, 1))
        )
        discriminator_A_loss_fake = discriminator_A.train_on_batch(
            fake_nontarget_data_use,
            np.zeros((BatchSize,) + (8, 1))
        )
        discriminator_A_loss = 0.5 * np.add(discriminator_A_loss_real, discriminator_A_loss_fake)

        discriminator_B_loss_real = discriminator_B.train_on_batch(
            real_target_data_use,
            np.ones((BatchSize,) + (8, 1))
        )
        discriminator_B_loss_fake = discriminator_B.train_on_batch(
            fake_target_data_use,
            np.zeros((BatchSize,) + (8, 1))
        )
        discriminator_B_loss = 0.5 * np.add(discriminator_B_loss_real, discriminator_B_loss_fake)

        discriminator_loss = 0.5 * np.add(discriminator_A_loss, discriminator_B_loss)

        # Train the generators
        generator_loss = combined_model.train_on_batch(
            [
                real_nontarget_data_use, real_target_data_use
            ],
            [
                np.ones((BatchSize,) + (8, 1)), np.ones((BatchSize,) + (8, 1)),
                real_nontarget_data_use, real_target_data_use
            ]
        )

        # if epoch % 100 == 0:
        #     print(f"{epoch} "
        #           f"[Discriminator loss: {discriminator_loss:.4f}] "
        #           f"[Generator loss: {generator_loss[0]:.4f}, "
        #           f"Validity_B loss: {generator_loss[1]:.4f}, "
        #           f"Validity_A loss: {generator_loss[2]:.4f}, "
        #           f"ReconNonTarget loss: {generator_loss[3]:.4f}, "
        #           f"ReconTarget loss: {generator_loss[4]:.4f}]")

    nontarget_idx = np.random.randint(0, real_nontarget_data.shape[0], augment_data)
    real_nontarget_data_transform = real_nontarget_data[nontarget_idx]
    augment_target_data = generator_AB.predict(real_nontarget_data_transform)

    # print(f"Training completed in {(time.time() - StartTime) / 60:.2f} minutes.\n")

    return np.array(augment_target_data)
