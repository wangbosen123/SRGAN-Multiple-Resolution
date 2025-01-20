import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, Add, LeakyReLU, Flatten, Dense, BatchNormalization, MaxPooling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from preprocess_data import *

options = BaseOptions()
options = options.initialize()
shape = options.size
shape = (int(shape), int(shape), 3)

def res_block(x, filters, kernel_size=3, strides=1, padding='same'):
    shortcut = x
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.4)(x)

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)

    x = Concatenate()([x, shortcut])
    x = LeakyReLU()(x)

    return x



def generator():
    inputs = Input(shape=shape)

    c1 = res_block(inputs, 64)  # 64 filters
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = res_block(p1, 128)  # 128 filters
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = res_block(p2, 256)  # 256 filters
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = res_block(p3, 512)  # 512 filters
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = res_block(p4, 1024)  # 1024 filters

    u6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(c5)
    u6 = Concatenate()([u6, c4])  # skip connection
    c6 = res_block(u6, 512)

    u7 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(c6)
    u7 = Concatenate()([u7, c3])  # skip connection
    c7 = res_block(u7, 256)

    u8 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(c7)
    u8 = Concatenate()([u8, c2])  # skip connection
    c8 = res_block(u8, 128)

    u9 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(c8)
    u9 = Concatenate()([u9, c1])  # skip connection
    c9 = res_block(u9, 64)

    outputs = Conv2D(3, kernel_size=1, activation='sigmoid')(c9)  # For binary segmentation

    model = Model(inputs, outputs)
    model.summary()
    return model



def discriminator():
    inputs = Input(shape=shape)

    # PatchGAN style discriminator
    x = Conv2D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x =  LeakyReLU(0.2)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    discriminator = Model(inputs, outputs, name='discriminator')
    discriminator.summary()
    return discriminator



if __name__ == '__main__':
    generator()
    discriminator()