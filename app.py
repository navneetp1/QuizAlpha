from flask import (Flask, render_template, request, redirect, url_for, session)
from bidict import bidict
from random import choice
import numpy as np
# bidict - both way dictionary
from tensorflow import keras
from stats.dataDist import createPlot
import os
import matplotlib


matplotlib.use('Agg')
 

ENCODER = bidict({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
                  'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16,
                  'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
                  'Y': 25, 'Z': 26})

app = Flask(__name__)
app.secret_key = 'abcdefg'

STATIC_DIR = os.path.join(app.static_folder, 'images')
os.makedirs(STATIC_DIR, exist_ok=True)


# overview of app

@app.route('/')
def index():
    session.clear()
    return render_template("index.html")

@app.route('/plot')
def plot():
    dataPath = os.path.join('data', 'keys.npy')
    fileName = 'barplot.png'
    plotPath = os.path.join(STATIC_DIR, fileName)

    createPlot(dataPath, plotPath)

    return render_template('bar.html', url=f"images/{fileName}")



# 1. Add data to be used for training - get post
@app.route("/add-data", methods=['GET'])
def getData():
    message = session.get('message', '')

    # in order instead of randomizing to get all 26 classes
    keys = np.load('data/keys.npy')
    count = {k:0 for k in ENCODER.keys()}
    for k in keys:
        count[k] += 1
    count = sorted(count.items(), key=lambda x: x[1])
    letter = count[0][0]    
    
    # letter = choice(list(ENCODER.keys()))
    
    # draw 'g'
    # g [0,0,0,0]
    # letters-labels : value-images
    return render_template('addData.html', letter=letter, message=message) # passing letter

@app.route("/add-data", methods=['POST'])
def storeData():
    key = request.form['letter']
    # saving the old and new labels by appending one after the other
    keys = np.load('data/keys.npy')
    keys = np.append(keys, key)
    np.save("data/keys.npy", keys)


    pixels = request.form['pixels']
    pixels = pixels.split(',')
    img = np.array(pixels).astype(float).reshape(1, 50, 50)
    imgs = np.load('data/images.npy')
    imgs = np.vstack([imgs, img])
    np.save("data/images.npy", imgs)

    session['message'] = f"{key} added to training dataset"

    return redirect(url_for('getData'))


# 2. prediction practice - get post
@app.route("/practice", methods=['GET'])
def practiceGetData():

    letter = choice(list(ENCODER.keys()))

    return render_template('practice.html', letter=letter, isCorrect='')

@app.route("/practice", methods=['POST'])
def practiceStoreData():

    letter = request.form['letter']

    pixels = request.form['pixels']
    pixels = pixels.split(',')
    img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

    model = keras.models.load_model('alphabet.keras')

    predicted_letter = np.argmax(model.predict(img), axis = -1)
    predicted_letter = ENCODER.inverse[predicted_letter[0]]

    isCorrect = 'YES' if predicted_letter == letter else 'NO'

    letter = choice(list(ENCODER.keys()))

    return render_template('practice.html', isCorrect=isCorrect, letter=letter)

if __name__ == "__main__":
    app.run(debug=True)