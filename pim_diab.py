# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:53:04 2020

@author: Mrunal
"""

import numpy as np
import pandas as pd
import warnings
import pickle
import os

#loading the dataset
data=pd.read_csv('pima_indian_diabetes.csv')

#Performing EDA
X=data.iloc[:, :-1].values
y=data.iloc[:, 7].values

#Feature selection
from sklearn.feature_selection import SelectKBest
selec=SelectKBest(k=3)
selec=selec.fit(X, y)
X=selec.transform(X)
cols_selec=selec.get_support()

#Train-test
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.02)

'''
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
X_train=mm.fit_transform(X_train)
X_test=mm.fit_transform(X_test)
'''
#SInce I'm using Naive Bayes, which is equipped for handling weight, no need of feature scaling

#Fitting model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)
'''
inp=[]
n=0
print("Glucose level is a value between 45-200, ")
while n<=3:
    lol=float(input("Enter glocose level, insulin level and age: "))
    inp.append(lol)
    n+=1
    continue

inp=np.array([inp])
#inp=mm.fit_transform(inp)
print(classifier.predict(inp))
'''

#print(classifier.predict([[0.136, 0.55, 0.065]]))

#Saving model
pickle.dump(classifier, open('model.pkl', 'wb'))
#loading the model
model=pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 9, 6]]))
