
from fileinput import filename
from flask import Flask, redirect, render_template, request, url_for, flash
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os


def model_predict(filename, loaded_model):
    image = load_img('uploads/'+filename)
    classes = ['Speed limit (20km/hr)', 'Speed limit (30km/hr)', 'Speed limit (50km/hr)',
               'Speed limit (60km/hr)', 'Speed limit (70km/hr)', 'Speed limit (80km/hr)',
               'End of speed limit (80km/hr)', 'Speed limit (100km/hr)', 'Speed limit (120km/hr)',
               'No passing', 'No passing for vehicles over 3.5 metric tons',
               'Right-of-way at the next intersection', 'Priority road', 'Yield',
               'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
               'No entry', 'General caution', 'Dangerous curve to the left',
               'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
               'Road narrows to the right', 'Road work', 'Traffic signals', 'Pedestrians',
               'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
               'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
               'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
               'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
               'End of no passing by vechiles over 3.5 metric tons']

    image = image.resize((50, 50))
    img = np.array(image)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    prediction = loaded_model.predict(img)

    idx = np.argmax(prediction[0])
    probability = prediction[0][idx]

    return classes[idx], probability*100, str(idx)


app = Flask(__name__)
app.secret_key = "secret key"
model = tf.keras.models.load_model('my_model')
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, acc, idx = model_predict(filename, model)
            flash(label)
            flash(acc)
            flash(idx)
            return redirect('/predict')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')


if __name__ == '__main__':

    app.run()
