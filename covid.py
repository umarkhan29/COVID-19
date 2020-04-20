import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, AvgPool2D, MaxPool2D
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array


#Enable Logging
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


#Defining dataset directory path
DATASET_DIR = "./dataset/"
normal_images = []
for img_path in glob.glob(DATASET_DIR + 'normal-images/*'):
	img = mpimg.imread(img_path)
	normal_images.append(img)

covid_images = []
for img_path in glob.glob(DATASET_DIR + 'covid/*'):
	img = mpimg.imread(img_path)
	covid_images.append(img)


print(len(normal_images))
print(len(covid_images))


#===================================================================================

#Defining Model and Training Parameters
INPUT_SHAPE = (255, 255, 3) #Using 255x255 RGB image
NB_CLASSES = 2
EPOCHS = 55
BATCH_SIZE = 5


#Building Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(250,(3,3)))
model.add(Activation("relu"))
  
model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2,2))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2,2))

model.add(Conv2D(256,(2,2)))
model.add(Activation("relu"))
model.add(MaxPool2D(2,2))
    
model.add(Flatten())
model.add(Dense(32))
#model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation("sigmoid"))

# Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Resizing Training data images to 255 x 255
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(255, 255),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training')



#Training the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    epochs = EPOCHS)


#====================================================
#Saving the model
model.save("my_covid_model.h5")
print("Model Saved !")

#==========================================================
#Accuracy
print("training_accuracy", history.history['accuracy'][-1])

#=================================================================

print(history.history)
#=================================================================
#Plotting Accuracy and Loss
#Plotting Accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#==========================================================================