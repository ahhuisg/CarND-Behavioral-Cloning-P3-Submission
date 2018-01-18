# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/img1.png "Image"
[image2]: ./examples/img1_flipped.png "Flipped Image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (the size of model.h5 is more than 500 MB, which cannot be checked into github. I put it into Google Storage. Here is the [Link](https://storage.googleapis.com/udacity-term1/model.h5))
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the implementation of NVIDIA's "End to End Learning for Self-Driving Cars" [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The model includes many RELU layers to introduce nonlinearity, and the data is normalized before entering the model.


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 102-106). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving as well as images from the left and right cameras.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make use of the images of the successful runs as much as possible, in order to clone and mimic the successful behavior (Behavior Cloning)

My first step was to make use of well-known pre-trained framework like VGG19. The output of the VGG19 model's 3rd layer from last was used together with a few manualy-added layers after that. The pre-trained model's layers up to the 5th layer from the last were freezed from training. In other words, only the weights of the last few layers as well as the few manually-added layers were learned, the first dozen or so layers of the VGG19 pre-trained model were freezed. In addition, I need to resize the input image be the shape of (224, 224, 3) as what VGG19 expected. However, The result was not as good expected. 


My second step was to make use of NVIDIA's "End to End Learning for Self-Driving Cars" [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Belows is the summary of the model architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Convolution 5x5     	| 24 filters, 2x2 stride, same padding	|
| RELU					|												|
| Max pooling	      	| 1x1 stride, 2x2 pool size			|
| Convolution 5x5     	| 36 filters, 2x2 stride, same padding|
| RELU					|												|
| Max pooling	      	| 1x1 stride, 2x2 pool size				|
| Convolution 5x5     	| 48 filters, 1x1 stride, same padding |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256					|
| Convolution 3x3     	| 64 filters, 1x1 stride, same padding	|
| RELU					|												|
| Max pooling	      	| 1x1 stride, 2x2 pool size					|
| Convolution 3x3     	| 64 filters, 1x1 stride, same padding	|
| RELU					|												|
| Max pooling	      	| 1x1 stride, 2x2 pool size					|
| Fully Connected 		| Output = 1164 					|
| RELU					|												|
| Fully Connected 		| Input = 1164, Output = 100 					|
| RELU					|												|
| Fully Connected 		| Input = 100, Output = 50 					|
| RELU					|												|
| Fully Connected 		| Input = 50, Output = 10 					|
| RELU					|												|
| Fully Connected 		| Input = 10, Output = 1 					|


#### 3. Creation of the Training Set & Training Process

I made use of the provided data set, which has already 8036 center lane images.

I also make use of the images from the left and right cameras, for whose steering angle I added 0.22 for the left images and minus 0.22 for the right ones.

To augment the data sat, I also flipped images and angles thinking that this would increase the size of the training data set. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

After the collection process, I had 48216 (8036 * 3 * 2) number of data points. I then preprocessed this data by normalizing the images' pixel values.

I finally randomly shuffled the data set and put around 5% (2056) of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the fact that both the training and valiation loss stop decreasing at epoch 8. I used an adam optimizer so that manually training the learning rate wasn't necessary.
