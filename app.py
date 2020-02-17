import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
Loan = pickle.load(open('Loan.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = Loan.predict(final_features)

    result = prediction[0]
    if (result==0):
        
        output = 'eligible for Loan'
    
    else :
    
         output = 'not eligible for Loan'
    
    return render_template('index.html', prediction_text='You are {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = Loan.predict([np.array(list(data.values()))])

    result = prediction[0]
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)