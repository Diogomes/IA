#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:22:31 2022

@author: me
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

n = 100
mse_list = []
seed_list = np.arange(0, n)

for seed in seed_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    mse_list.append(mean_squared_error(y_test,y_pred))
    
plt.plot(seed_list, mse_list)
plt.title("MSE VS RANDOM STATE")
plt.xlabel('random state')
plt.ylabel('mse')
plt.grid()
plt.savefig('mse vs random state.png')
plt.show()