# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:11:58 2018

@author: Anirban
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import csv

dataset = pd.read_csv('Google_Stock_Price_Train.csv')
dataset

dates = []

prices = []

def get_file(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) #skkiping columns
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[0]))
            prices.append(float(row[1]))
        return
    
def predicted_price(dates, prices, x):
        dates = np.reshape(dates,(len(dates), 1)) #converting matrix of n x 1
        
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma= 0.1)
        svr_lin = SVR(kernel='linear', C=1e3)
        svr_poly = SVR(kernel='poly', C=1e3, degree= 2)
        
        svr_rbf.fit(dates, prices) #fitting the data points in models
        svr_lin.fit(dates, prices)
        svr_poly.fit(dates, prices)
        
        plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 

        plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	    
        plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	    
        plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	    
        plt.xlabel('Date')
	    
        plt.ylabel('Price')
	    
        plt.title('Support Vector Regression')
	    
        plt.legend()
	    
        plt.show()
        
        return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
    
get_file('Google_Stock_Price_Train.csv')  # calling get_file method by passing the csv file to it
print('Dates', dates)
print('Prices', prices)

predicted_price = predicted_price(dates, prices, 24)
print('Predicted price for 29-May is:')
print ("RBF kernel: $", str(predicted_price[0]))
print ("Linear kernel: $", str(predicted_price[1]))
print ("Polynomial kernel: $", str(predicted_price[2]))

