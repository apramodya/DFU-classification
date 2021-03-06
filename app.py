import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

# ::: Prepare pre-aug model :::
PRE_AUG_MODEL_WEIGHTS = './_models/pre-aug/classifier.h5' 
PRE_AUG_MODEL_ARCHITECTURE = './_models/pre-aug/model.json'

json_file = open(PRE_AUG_MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
pre_aug_model = model_from_json(loaded_model_json)
pre_aug_model.load_weights(PRE_AUG_MODEL_WEIGHTS)

# ::: Prepare post-aug model :::
POST_AUG_MODEL_WEIGHTS = './_models/post-aug/classifier.h5' 
POST_AUG_MODEL_ARCHITECTURE = './_models/post-aug/model.json'

json_file = open(POST_AUG_MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
post_aug_model = model_from_json(loaded_model_json)
post_aug_model.load_weights(POST_AUG_MODEL_WEIGHTS)

# ::: Model prediction :::
def model_predict(filename, model1, model2):
    basepath = os.path.dirname(__file__)
    directory_path = os.path.join(basepath, 'static/img/uploads')
    file_path = os.path.join(directory_path, filename)

    image = cv2.imread(file_path)
    image = cv2.resize(image, (128, 128))
    image =  np.array([image])
    image = image / 255.0

    pre_aug_pred = model1.predict(image)[0][0]
    post_aug_pred = model2.predict(image)[0][0]

    return (pre_aug_pred, post_aug_pred)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        f = request.files['file']

        if f.filename == '':
            return render_template('home.html')

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/img/uploads', secure_filename(f.filename))
        f.save(file_path)

        result = model_predict(f.filename, pre_aug_model, post_aug_model)
        
        preAugResult =  "{:.2f}".format(result[0] * 100)
        _preAugResult =  "{:.2f}".format((1 - result[0]) * 100)
        postAugResult = "{:.2f}".format(result[1] * 100)
        _postAugResult = "{:.2f}".format((1 - result[1]) * 100)

        return render_template('home.html', preAugResult = preAugResult, _preAugResult = _preAugResult, postAugResult = postAugResult, _postAugResult = _postAugResult)

@app.route("/about")
def about():
    return render_template('about.html')

# ::: Run main :::
if __name__ == "__main__":
    app.run(debug=True)