import sys
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder='templates')

model = pickle.load(open('brain_stroke3.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['age','avg_glucose_level','bmi','hypertension','heart_disease','ever_married','work_type','smoking_status']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output[0] == 1:
        res_val = "Stroke Can occur"
    else:
        res_val = "Stroke won't occur"

    return render_template('index.html', prediction= ' {}'.format(res_val))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8000)