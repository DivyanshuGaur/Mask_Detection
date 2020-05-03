from flask import Flask,request,jsonify

from tensorflow.keras.models import load_model
import joblib
from flask_cors import CORS

import os
import json

from absl import logging
logging._warn_preinit_stderr = 0
import numpy as np

from tensorflow.keras.preprocessing import image



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app=Flask(__name__)


CORS(app)

model=load_model('Mask_Classifier.h5')



@app.route("/")

def index():
    return "<h1>  Flask  Started </h1>"




@app.route('/frontend',methods= ['POST'])
def frontend():
    a=request.files
    print(a)

    isthisFile = request.files.get('file')
    print(isthisFile)
    print(isthisFile.filename)
    isthisFile.save("./" + isthisFile.filename)
    cls=return_class(isthisFile.filename)
    return jsonify(ans=cls)




def return_class(bimg):
    myimage = image.load_img(bimg, target_size=(160,160,3))
    myimage_arr = image.img_to_array(myimage)
    print(myimage_arr.shape)
    myimage_arr = np.expand_dims(myimage_arr, axis=0)
    print(myimage_arr.shape)

    pred = model.predict_classes(myimage_arr)[0]

    print(pred)

    li = ['mask', 'non-mask']

    return (li[pred[0]])


if __name__ == '__main__':
    app.run()