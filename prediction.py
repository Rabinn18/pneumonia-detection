import tensorflow as tf
import gradio as gr
import cv2
import numpy as np

from tensorflow.keras.utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation

def swish_activation(x):
        return (K.sigmoid(x) * x)

get_custom_objects().update({'swish_activation': Activation(swish_activation)})
# model = tf.keras.models.load_model('weights.hdf5')
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
model = load_model('detectionmodel.h5')


def preprocess_image(image):
    image = cv2.resize(image,(150,150))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image) / 256
    image = np.expand_dims(image, axis=0)
    return image

def predpneumoniaPrediction(image):
    # Preprocess the image
    image= preprocess_image(image)
    # Make a prediction using the model
    model_pred = model.predict(image)
    probability = model_pred[0]
    if probability[0] < 0.5:
        result = 'Pneumonia POSITIVE. Consult Radiologist as soon as possible.'
    else:
        result = 'Pneumonia NEGATIVE. You have a Healthy lung.'
        
   
    # Return the predicted class and processed images
    return result
# def predpneumoniaPrediction(image):
#     # Preprocess the image
#     image= preprocess_image(image)
#     # Make a prediction using the model
#     isPneumonic = model.predict(image)
#     probability = isPneumonic
#     imgClass = "Normal" if isPneumonic.any()<0.5 else "Pneumonic"
#     return imgClass

Pneumonia_interface = gr.Interface(predpneumoniaPrediction,
                     inputs=gr.inputs.Image(), 
                     outputs=["text"],
                     title="Pneumonia Prediction CNN model",
                     description="Upload an image and the model")

Pneumonia_interface.launch()
