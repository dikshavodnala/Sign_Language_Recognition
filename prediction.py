import cv2
import numpy as np
from variables import *

import tensorflow as tf
model = tf.keras.models.load_model(model_path)


def preprocess_image(img_array):
    # Convert RGB to grayscale
    img_array = cv2.cvtColor(img_array , cv2.COLOR_BGR2GRAY)
    # Resize the image to match the model's expected input size
    img_array  = cv2.resize(img_array, (100, 100))
    img_array = img_array.reshape(image_size,image_size,1)
    
    # Normalize the pixel values
    img_array = img_array / 255.0
    # Reshape the image to add batch dimension and single channel
    img_array = np.expand_dims(img_array,axis=0)
    return img_array

def predict(img_array):
    img_array = preprocess_image(img_array)
    preds = model.predict(img_array)
    preds = preds*100
    max_index = int(np.argmax(preds))
    return preds.max(), labels[max_index]




