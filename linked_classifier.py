# A classifier that identifies the type of microfluidic chip being analyzed and determines
# whether or not the chip image contains an outlet

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
from keras.models import load_model
from PIL import ImageFont
from PIL import ImageDraw 

# load data
prediction_data = os.path.normpath(os.getcwd()+"/linked_classifier_predictions/predictions")
imageski = prediction_data+os.path.normpath('/predict/IMG1448.png')

# load models
outlet_model = load_model('outlet_model.h5')
chip_model = load_model('chip_model.h5')

img_width, img_height = 100, 500 # resized image dimensions

predict_datagen = ImageDataGenerator(rescale = 1. / 255)
predict_batch = predict_datagen.flow_from_directory(
    prediction_data,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='binary'
)

def predict_image(mod, img):
    img = load_img(img)
    img = img.resize((500, 100), Image.ANTIALIAS)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    return mod.predict_classes(x)

# determines the type of chip and whether or not an outlet is visible
if outlet_model.predict_generator(predict_batch, steps=1) < 0.5:
    print('Contains Outlet')
    if predict_image(chip_model, imageski) == 0:
        decision = 'Chip A Outlet'
        print(decision)
    if predict_image(chip_model, imageski) == 1:
        decision = 'Chip B Outlet'
        print(decision)
    if predict_image(chip_model, imageski) == 2:
        decision = 'Chip C Outlet'
        print(decision)
else:
    print('No Outlet')
    if predict_image(chip_model, imageski) == 0:
        decision = 'Chip A No Outlet'
        print(decision)
    if predict_image(chip_model, imageski) == 1:
        decision = 'Chip B No Outlet'
        print(decision)
    if predict_image(chip_model, imageski) == 2:
        decision = 'Chip C No Outlet'
        print(decision)

# appends the prediction decision text to the image
def append_decision(img):
    font = ImageFont.load_default()
    img = Image.open(img)
    draw = ImageDraw.Draw(img)
    draw.text((0,0), decision, (255, 255, 0), font=font)
    img.show()

append_decision(imageski)
