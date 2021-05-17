from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# import the libraries as shown below
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
tf.keras.applications.mobilenet.preprocess_input
#from tensorflow.keras.applications.resnet152V2 import ResNet152V2
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plttf.__version__

import pathlib
dataset_url = "C:/Users/Dheeraj Kumar/Downloads/Downloads/Deep-learning/crop-dataset/train"
test_url ="C:/Users/Dheeraj Kumar/Downloads/Downloads/Deep-learning/crop-dataset/test"

batch_size = 32
img_height = 180
img_width = 180

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# re-size all the images to this
IMAGE_SIZE = [180, 180]

import tensorflow
MobileNet =tensorflow.keras.applications.MobileNet(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in MobileNet.layers:
    layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(MobileNet.output)

prediction = Dense(5, activation='softmax')(x)

# create a model object
model = Model(inputs=MobileNet.input, outputs=prediction)

model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(dataset_url,
                                                 target_size = (180,180),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(test_url,
                                            target_size = (180, 180),
                                            batch_size = 32,
                                            class_mode = 'categorical')
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

from tensorflow.keras.models import load_model

model.save('C:/Users/Dheeraj Kumar/Downloads/Downloads/Crop-Classification/models/mobileNet')
y_pred = model.predict(test_set)
import numpy as np
y_pred = np.argmax(y_pred, axis=1)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('C:/Users/Dheeraj Kumar/Downloads/Downloads/Crop-Classification/models/mobileNet')

# !pip install flask_ngrok


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='C:/Users/Dheeraj Kumar/Downloads/Downloads/Crop-Classification/models/mobileNet'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(180,180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Jute"
    elif preds==1:
        preds="Maize"
    elif preds==2:
        preds="Rice"
    elif preds==3:
        preds="Sugarcane"
    else:
        preds="Wheat"
        
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f=request.files['file']
        file_path=os.path.join("C:/Users/Dheeraj Kumar/Downloads/Downloads/Crop-Classification/uploads" ,secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
 
  app.run()
# threading.Thread(target=app.run, kwargs={'host':'127.0.0.1','port':5001}).start()

    