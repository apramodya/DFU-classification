import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

app = Flask(__name__)

# ::: Prepare pre-aug model :::
PRE_AUG_MODEL_WEIGHTS = './models/pre-aug/classifier.h5' 
PRE_AUG_MODEL_ARCHITECTURE = './models/pre-aug/model.json'

json_file = open(PRE_AUG_MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(PRE_AUG_MODEL_WEIGHTS)

# ::: Prepare post-aug model :::
POST_AUG_MODEL_WEIGHTS = './models/post-aug/classifier.h5' 
POST_AUG_MODEL_ARCHITECTURE = './models/post-aug/model.json'

json_file = open(POST_AUG_MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(POST_AUG_MODEL_WEIGHTS)

# ::: Model prediction :::
def model_predict(img_path, model):
    img_normal = image.load_img(img_path, target_size=(128, 128))
    img_normal = image.img_to_array(img_normal)/255
    img_normal = np.array([img_normal])
    pred = model.predict(img_normal)

    return pred

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/img/uploads', secure_filename(f.filename))
        f.save(file_path)

        preAugResult = "1"
        postAugResult = "2"
        return render_template('home.html', preAugResult = preAugResult, postAugResult = postAugResult)

# ::: Run main :::
if __name__ == "__main__":
    app.run(debug=True)