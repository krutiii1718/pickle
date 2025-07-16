from flask import Flask, render_template, request
import pickle
import numpy as np

app1 = Flask(__name__)
model = pickle.load(open('Model.pkl','rb'))

@app1.route('/')
def home():
    return render_template('index.html')

@app1.route('/predict', method=['POST'])
def predict():
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    sqft = float(request.form['sqft_living'])

    features = np.array([[bedrooms, bathrooms, sqft]])
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction_text=f'Estimate price:')

if __name__ == '__main__':
    app.run(debug=True)