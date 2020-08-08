# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 23:59:57 2020

@author: Admin
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feat=[float(x) for x in request.form.values()]
    final_feat= [np.array(int_feat)]
    prediction=model.predict(final_feat)
    
    output=prediction
    
    return render_template('index.html', pred_text="{} , 0= No diabetes(negative), 1=Diabetes positive".format(output))

if __name__=="__main__":
    app.run(debug=True)