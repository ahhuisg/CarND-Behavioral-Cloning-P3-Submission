import numpy as np
import matplotlib.image as mpimg
from scipy.misc import imresize
import random
import pandas as pd

img_dir = 'data/IMG/'
df = pd.read_csv('data/driving_log.csv')
size = len(df)

steering_adj = 0.22

def total_num_data_rows():
    return size

def add_image(file_name, measurement, v_adj): 
    image = mpimg.imread(img_dir + file_name.split('/')[-1])
    return image, measurement + v_adj

def generate_batch(batch_size=64):
    """
        Generator to generate next batch data for training and validation
        Each batch consists of batch_size number of rows
        Each row contains 3 images from center, left and right respectively
        A steering adjustment value is applied for left and right images
    """
    while True:
        images = []
        measurements = []
        indices = random.sample(range(size), batch_size)

        for ind in indices:
            row = df.iloc[ind]

            t1 = add_image(row['center'], row['steering'], 0)
            images.append(t1[0])
            measurements.append(t1[1])

            t2 = add_image(row['left'], row['steering'], steering_adj)
            images.append(t2[0])
            measurements.append(t2[1])

            t3 = add_image(row['right'], row['steering'], -steering_adj)
            images.append(t3[0])
            measurements.append(t3[1])

        x_train = np.array(images)
        y_train = np.array(measurements)

        x_train = x_train/255 - 0.5
        x_train, y_train = flip(x_train, y_train)

        yield x_train, y_train


def flip(x_train, y_train):
    #Flip images and add to training set

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
    # reside input image to be the size that VGG19 model expects

    images = []
    measures = []
    
    for image, measure in zip(x_train, y_train):
        img1 = imresize(image, (224, 224))
        
        images.append(img1)
        measures.append(measure)
        
    return np.array(images), np.array(measures)
