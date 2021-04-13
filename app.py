from flask import Flask, request
import joblib
import numpy as np
import re
import os

app = Flask(__name__)

@app.route('/')
def helloworld():
    return 'Hello Welcome to Iris Classification API model plz path /iris and POST param: value1,value2,value3,value4,'

@app.route('/iris', methods=['POST'])
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['param']
    inputs = np.array(req.split(','), dtype=np.float32).reshape(1, -1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Sentosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virginica'           

if __name__== '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)