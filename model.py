from flask import Flask, render_template,request
from tensorflow import keras
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
import keras
from tensorflow.keras import datasets,layers,models

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout 
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from tensorflow import keras
app = Flask(__name__)
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_json(loaded_model_yaml)
# load weights into new model
model.load_weights("model.h5")


@app.route('/')
def index():
    return render_template("html.html", name="Tariq")


@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    img.save('img.jpg')
    

    img = image.load_img("img.jpg", target_size=(224,224))
    x=image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x,axis=0)
    images=np.vstack([resized_img_np])
    prediction = model.predict(images)
    if prediction[0][0]<0.5:
         pred = 'Potholes'
    else:
         pred="Normal"

    
      


    return render_template("prediction.html", data=pred)


if __name__ =="__main__":
    app.run(debug=True)