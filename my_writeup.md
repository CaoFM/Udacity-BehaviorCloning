
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files

#### 1. Are all required files submitted?

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* my_writeup.md summarizing the results
* run1.mp4 is the video file showcase one lap around the track in autonomous mode

### Quality of Code

#### 1. Is the code functional?
Use model.py to train the model
```sh
python model.py
```

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 2. Is the code usable and readable?

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Here's a quick summary of the main steps:

- Load sample driving data csv file
- Load my driving data csv files if desired(stored in various folders for clarity)
- Build a generator to provide batch data
- Build a function to manipulate the data
- Build NN model
- Run training and save trained model

### Model Architecture and Training Strategy

#### 1. Has an appropriate model architecture been employed for the task?

My model starts with prepropessing layers, then followed by convolutional layers, and finally fully connected layers.

#### 2. Has an attempt been made to reduce overfitting of the model?

The model contains dropout layers in order to reduce overfitting. 

#### 3. Have the model parameters been tuned appropriately?

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Is the training data chosen appropriately?

Training data was chosen to keep the vehicle driving on the road. 
In addition to the sample driving data provided by Udacity, I recorded driving data from several different senarios:
- driving center of the road
- recover from the left side
- recover from the right side
- a difficult spot near bridge
- a difficult spot past bridge

### Architecture and Training Documentation

#### 1. Is the solution design documented?

I started with the most simple model with one conv layer and one fully connected layer. The goal was first to verify the pipeline works. Once the training complete, the car is wondering around the road and not driving well at all as expected.

Then I experimented with various combination of the following aspects to compare performance:
- I built more and more layers progressively and introduce drop out layers
- I change whether to include three cameras or just one camera
- I add the flipping of images
- I tried to shift the image horizontally

The performance however various with little improvement. I stuck for 10 GPU hours until I read a thread on the forum that the drive.py file receives RGB file while my model uses BGR via cv2.imread while training.

Once I added the convert to RGB in my training, the vehicle can suddenly drive very well around the simulator track.


#### 2. Is the model architecture documented?

The final model looks like this:

Lambda layer with input_shape 160,320,3
Cropping image with (75,25),(0,0)

Convolution Layer (24,(5,5)) with max pooling (2x2) with relu activation 
Convolution Layer (36,(5,5)) with max pooling (2x2) with relu activation 
Convolution Layer (48,(5,5)) with max pooling (2x2) with relu activation 
Convolution Layer (64,(3,3)) with max pooling (2x2) with relu activation 

Flatten
Fully connected layer Dense(100)
Dropout(0.2)
Fully connected layer Dense(50)
Fully connected layer Dense(1)


#### 3. Is the creation of the training dataset and training process documented?
I took notice of the provided sample driving data contains too many zero angle data. So when I load the data, I randomly left out 40% of the straight driving data.

To capture good driving behavior, I first recorded three laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would recover to the center if it ever drifts to the side.

These driving data are located in mydata/ folder.

In my function Augment_Data, I have experimented two approaches:
- randomly flipped images and angles
- randomly shifted image horizontally and compensate for an angle proportional to the amount of shifting.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

### Simulation

#### 1. Is the car able to navigate correctly on test data?

Video file run1.mp4. 