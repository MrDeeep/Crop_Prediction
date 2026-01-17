import os
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='website', static_folder='website')

model_path = os.path.join(os.path.dirname(__file__), 'ML_model', 'crop_prediction.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['Rainfall'])
    except (KeyError, ValueError):
        return render_template('index.html', prediction_text='Invalid input: please enter numeric values for all fields.')

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    output = prediction[0]
    return render_template('index.html', prediction_text=f'The Crop is {output}')

if __name__ == "__main__":
    app.run(debug=True)
