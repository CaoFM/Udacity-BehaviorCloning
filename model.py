import csv
import cv2
import numpy as np
import sklearn
from math import ceil

# Fill the 'Data' list of applicable (image,measurement) tuple
# Data = [(image1, measurement1),
#         (image2, measurement2),
#         ... ...
#        ]

Data = []

# parameters
correct_left = 0.2
correct_right = -0.2

# Load Udacity sample driving data

num_of_images = 0
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile) # Udacity csv has a header line
    next(reader, None)
    for line in reader:
        cam_center = '/opt/carnd_p3/data/'+line[0].strip()
        cam_left = '/opt/carnd_p3/data/'+line[1].strip()
        cam_right = '/opt/carnd_p3/data/'+line[2].strip()
        angle = float(line[3].strip())
        
        if (angle != 0) or (np.random.rand() > 0.4): # retain 60% of straight driving
            Data.append((cam_center,angle))
            Data.append((cam_left,angle+correct_left))
            Data.append((cam_right,angle+correct_right))
        
            num_of_images += 1

    print('%d images from sample driving' % num_of_images)

# Load my selected driving data
# Comment out for not using
# IMG location in recorded in absolute path

my_driving_log = [\
#                  './mydata/lap1/driving_log.csv',\
#                  './mydata/lap2/driving_log.csv',\
#                  './mydata/lap3/driving_log.csv',\
                  './mydata/recover_left/driving_log.csv',\
                  './mydata/recover_right/driving_log.csv',\
#                  './mydata/track2/driving_log.csv'\
                   './mydata/bridge/driving_log.csv',\
                  './mydata/past_bridge/driving_log.csv'\
                 ]

for log in my_driving_log:
    print ('Loading driving log from',log)
    
    num_of_images = 0
    
    with open(log) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip header just in case
        for line in reader:
            cam_center = line[0].strip()
            cam_left = line[1].strip()
            cam_right = line[2].strip()
            angle = float(line[3].strip())
            
            if 1:
                Data.append((cam_center,angle))
                #Data.append((cam_left,angle+correct_left))
                #Data.append((cam_right,angle+correct_right))
        
                num_of_images += 1
        
        print('%d frames driving data loaded' % num_of_images)
    
from sklearn.model_selection import train_test_split

# split samples into training and validation
# train_test_split performs shuffling by default
train_samples, validation_samples = train_test_split(Data, test_size=0.2)

print('%d training samples' % len(train_samples))
print('%d validation samples' % len(validation_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for each_sample in batch_samples:
                image_path = each_sample[0]              
                angle = each_sample[1]
                
                image,angle = Augment_Data(image_path, angle)
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)

def Augment_Data(image_path,angle):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # I heard that drive.py reads RGB
    
    # randomly flip the image
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        angle = -angle

    # randomly shift the image horizontally
    if 0: # for trial
        shift_range = 50
        shift = np.random.randint(shift_range) -(shift_range/2)
    
        m = np.float32([[1, 0, shift], [0, 1, 0]])
        rows, cols = image.shape[:2]
        image = cv2.warpAffine(image, m, (cols, rows))
    
        angle = angle + shift * (-0.02)
    
    return image,angle
    
            
batch_size = 32            
            
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)
   
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x /127.5 -1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Conv2D(24,(5,5))) #todo whats the parameter here
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.8))
model.add(Activation('relu'))
model.add(Conv2D(36,(5,5))) #todo whats the parameter here
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
#model.add(Dropout(0.8))
model.add(Conv2D(48,(5,5))) #todo whats the parameter here
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3))) #todo whats the parameter here
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
#model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,\
                    steps_per_epoch= ceil(len(train_samples)/batch_size),\
                    validation_data=validation_generator,\
                    validation_steps=ceil(len(validation_samples)/batch_size),\
                    epochs=2, verbose=1)

model.save('model.h5')

# model x good except for bridge
