import numpy as np
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Lambda

import csv
import cv2
from PIL import Image

images = []
measurements = []

#data_dirs = ['test1', 'test2', 'test3']
data_dirs = ['org']

counter = 0

def add_image(file_name, measurement):
    image = mpimg.imread(img_dir+'IMG/'+file_name)
    
    images.append(image)
    measurements.append(measurement)

    #image_flipped = np.fliplr(image)
    #images.append(image_flipped)
    #measurement_flipped = -measurement
    #measurements.append(measurement_flipped)

for dt in data_dirs:
    img_dir = 'data/'+dt+'/'
    
    print ('Processing data directory: '+img_dir)
    with open(img_dir+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if counter > 10:
                pass
                
            center_file_name = line[0].split('/')[-1]
            left_file_name = line[1].split('/')[-1]
            right_file_name = line[2].split('/')[-1]
            measure = float(line[3])
            add_image(center_file_name,measure)
            #add_image(left_file_name,(measure+0.2))
            #add_image(right_file_name,(measure-0.2))
            counter = counter + 1

x_train = np.array(images)
y_train = np.array(measurements)


from scipy.misc import imresize

def flip(x_train, y_train):
    images = []
    measures = []
    for image, measure in zip(x_train, y_train):
        image_flipped = np.fliplr(image)
        
        images.append(image)
        measures.append(measure)
        
        images.append(image_flipped)
        measures.append(-measure)
    
    return np.array(images), np.array(measures)

def crop(x_train, y_train):
    images = []
    measures = []
    
    for image, measure in zip(x_train, y_train):
        cropped_img = image[70:-25,]
        
        images.append(cropped_img)
        measures.append(measure)
        
    return np.array(images), np.array(measures)


def resize_vgg19(x_train, y_train):
    images = []
    measures = []
    
    for image, measure in zip(x_train, y_train):
        img1 = imresize(image, (224, 224))
        
        images.append(img1)
        measures.append(measure)
        
    return np.array(images), np.array(measures)


from keras.applications import VGG19
from keras.layers import Input, Dropout, BatchNormalization
from keras.models import Model

def cust_vgg19():
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
x_train = x_train/255 - 0.5
x_train, y_train = flip(x_train, y_train)

model = nvidia_model()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=10)
model.save('model_nvidia.h5')

