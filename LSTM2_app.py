import numpy as np
import pickle
from flask import Flask, request, render_template

# Create a Flask application

app = Flask(__name__, template_folder='templates')

# Load pickle model

model = pickle.load(open('LSTM2_history.pkl', 'rb'))


# Create a home page

@app.route('/')
def home():
    return render_template('LSTM2_index.html')

# Create a POST method

@app.route('/predict', methods = ['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    return render_template('LSTM2_index.html', prediction_text = 'Multivariate LSTM Prediction: Fe & Acidity should be {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug = True)

