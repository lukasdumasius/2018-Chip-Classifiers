# A classifier that determines whether a microfluidic chip is of a certain type
# Currently:
# CBC - 0
# DropMaker - 1
# LD 21 - 2

import os, sys
import keras #machine learning
from keras import backend as K
import numpy as np #math
import numpy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image

img_width, img_height = 100, 500 # resized image dimensions
nb_train_samples = 10000 # total
nb_validation_samples = 10000 # total
epochs = 10
batch_size = 20

# collect data
training_data = os.path.normpath(os.getcwd()+"/chipA_chipB_data/training")
validation_data = os.path.normpath(os.getcwd()+"/chipA_chipB_data/testing")
prediction_data = os.path.normpath(os.getcwd()+"/chipA_chipB_data/predictions")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# convolutional layers
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# fully connected layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))

# print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0,
    zoom_range=0.1,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    vertical_flip=False,
    horizontal_flip=False,
    fill_mode='nearest'
    )

img = load_img(training_data+os.path.normpath('/DropMaker/IMG1612.png'))  # this is a PIL image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# data augmentation example
i = 0
for batch in train_datagen.flow(x, 
                                batch_size=1, 
                                save_to_dir=os.getcwd()+os.path.normpath('/chipA_chipB_preview'), 
                                save_prefix='transformed', 
                                save_format='png'
                                ):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

test_datagen = ImageDataGenerator(
    rescale=1. / 255
    )

train_generator = train_datagen.flow_from_directory(
    training_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
    )

validation_generator = test_datagen.flow_from_directory(
    validation_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
    )

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
    ) 

# saves the model and its weights
model.save('chip_model.h5')
model.save_weights("chip_CNN.h5")

# prediction - outlet or no outlet
def predict_image(mod, img):
    img = load_img(img)
    img = img.resize((500, 100), Image.ANTIALIAS)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    print(mod.predict_classes(x))

# print("CBC Chip")
# predict_image(model, prediction_data+os.path.normpath('/CBC/IMG0882.png'))
# predict_image(model, prediction_data+os.path.normpath('/CBC/IMG0885.png'))
# predict_image(model, prediction_data+os.path.normpath('/CBC/IMG0935.png'))
# predict_image(model, prediction_data+os.path.normpath('/CBC/IMG1008.png'))
# predict_image(model, prediction_data+os.path.normpath('/CBC/IMG1081.png'))
# print("DropMaker Chip")
# predict_image(model, prediction_data+os.path.normpath('/DropMaker/IMG1403.png'))
# predict_image(model, prediction_data+os.path.normpath('/DropMaker/IMG1431.png'))
# predict_image(model, prediction_data+os.path.normpath('/DropMaker/IMG1443.png'))
# predict_image(model, prediction_data+os.path.normpath('/DropMaker/IMG1468.png'))
# predict_image(model, prediction_data+os.path.normpath('/DropMaker/IMG1643.png'))
# print("LD 21 Chip")
# predict_image(model, prediction_data+os.path.normpath('/LD21/IMG2520.png'))
# predict_image(model, prediction_data+os.path.normpath('/LD21/IMG2548.png'))
# predict_image(model, prediction_data+os.path.normpath('/LD21/IMG2577.png'))
# predict_image(model, prediction_data+os.path.normpath('/LD21/IMG2585.png'))
# predict_image(model, prediction_data+os.path.normpath('/LD21/IMG2519.png'))










