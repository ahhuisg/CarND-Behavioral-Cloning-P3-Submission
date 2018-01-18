from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.applications import VGG19
from keras.layers import Input, Dropout, BatchNormalization
from keras.models import Model

import data_helper as dh


def cust_vgg19():
    # Use VGG19 pretrain model. I freeze the weights of VGG19 up to the last 3 layers
    # Train the last 3 layers of VGG19 as well as a few more Dense layers with batch normalization and dropouts

    input_image = Input(shape = (224,224,3))

    base_model = VGG19(input_tensor=input_image, include_top=False)

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    x = base_model.get_layer("block5_conv4").output
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu")(x)
    x = Dense(2048, activation="relu")(x)
    x = Dense(1, activation="linear")(x)

    model = Model(input=input_image, output=x)
    return model

def nvidia_model():
    # Based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
    # https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    model = Sequential()

    #model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))

    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), input_shape=(160,320,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation("relu"))

    model.add(Dense(100))
    model.add(Activation("relu"))

    model.add(Dense(50))
    model.add(Activation("relu"))

    model.add(Dense(10))
    model.add(Activation("relu"))

    model.add(Dense(1))

    return model


print ('Starting training with NVIDIA model')

model = nvidia_model()
model.compile(loss='mse', optimizer='adam')

# Create data generator for training and validation
train_generator = dh.generate_batch()
validate_generator = dh.generate_batch()

# Number of samples to train for each epoch
total_num_samples = dh.total_num_data_rows() * 6

history = model.fit_generator(train_generator,
                              samples_per_epoch=total_num_samples,
                              nb_epoch=10,
                              validation_data=validate_generator,
                              nb_val_samples=2560)

model.save('model.h5')