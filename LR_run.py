#!/usr/bin/python

from random import seed
from random import randrange
from csv import reader
from math import sqrt
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo




#Define Sigmoid
def sig(z):
	g = list()
	for i in z:
		g.append(1/(1+exp(-i)))
	return np.array(g)

#Define Cost Function Return Cost and Gradient
def costFunction(theta, X, Y, lam):
	m = float(len(Y))
	H = sig(np.dot(X,theta)) 
	J = (1/m)*sum(-np.dot(Y,log(H)) - np.dot(1-Y,log(1-H)))+(lam/(2*m))*sum(theta*theta) 
	return J

def costGrad(theta, X, Y, lam):
	m = float(len(Y))
	H = sig(np.dot(X,theta)).flatten()  
	dJ = np.zeros(theta.shape)
	dJ[0] = (1/m)*(np.dot(X[:,0],(H-Y)));
	dJ[1:] = (1/m)*(np.dot(X[:,1:].transpose(),(H-Y))) + (lam/m)*theta[1:];
	return dJ

#Optimize classifier using spo.fmin_cg
def train(X,Y,lam):
	eps = 0.001
	theta = np.zeros((shape(X)[1],1))#eps*rand(shape(X)[1],1)
	result = spo.fmin_cg(costFunction,fprime = costGrad, x0 = theta, args=(X,Y,lam),maxiter=200)
	return result

#Train using steepest descent
def train_stp_dec(X,Y,lam):
        theta   = np.zeros((shape(X)[1],1)).flatten()
        eps     = 1e-5  #minimal improvement
        LR      = 0.1
        (m,n)   = X.shape
        costOld = 0
        cost    = costFunction(theta,X,Y,lam)
        #print('Cost', cost)
        cnt     = 0
        while(abs(cost - costOld) > eps):
                theta = theta - LR*costGrad(theta,X,Y,lam)
                costOld = cost
                cost = costFunction(theta,X,Y,lam)
                cnt = cnt+1
                #print('Cost', cost)

        return theta


#Predict
def predict(theta,X): #Only takes in single X row
	z =  np.dot(X,theta) 
	H = 1/(1+exp(-z))
	return H>0.5



# Calculate and plot learning curve
def learningCurve(X,y,X_val,y_val,lam):
	m = shape( X )[0]
	error_train = list()
	error_val = list()

	m_val = shape( X_val )[0]

	for i in range( 0, m ):
		theta = train( X[0:i+1,:], y[0:i+1], lam )
		error_train.append( costFunction( theta, X[0:i+1,:], y[0:i+1], lam ) )
		error_val.append( costFunction( theta, X_val, y_val, lam ) )

	error_train = array(error_train)
	error_val   = array(error_val)

	# number of training examples
	temp = np.ones((m,1))

	plt.ylabel('Error')
	plt.xlabel('Number of training examples')
	#plt.ylim([0, 5])
	#plt.xlim([1, m])
	plt.plot( temp, error_train, color='b', linewidth=2, label='Train' )
	plt.plot( temp, error_val, color='g', linewidth=2, label='Cross Validation' )
	plt.legend()
	plt.show() #block=True )

	return error_train, error_val



def validationCurve( X, y, X_val, y_val, lamda_vec ):
        error_train = []
        error_val       = []

        for lamda in lamda_vec:
		#NOTE THAT FMINCG CAN HAVE RUNAWAY PROBLEMS WITH LOW LAMBDA
                #theta = train_stp_dec( X, y, lamda )
                theta = train( X, y, lamda )

                error_train.append( costFunction( theta, X, y, lamda ) )
                error_val.append( costFunction( theta, X_val, y_val, lamda ) )

        error_train = array( error_train )
        error_val       = array( error_val )

        plt.ylabel('Error')
        plt.xlabel('Lambda')
        plt.plot( lamda_vec, error_train, 'b', label='Train' )
        plt.plot( lamda_vec, error_val, 'g', label='Cross Validation' )
	#plt.xscale( 'log' )
        #plt.text( -15, 45, 'Validation curve' )
        plt.legend()
        plt.show()# block=True )

        return error_train, error_val


#Create polynomial (degree p+1) features
def polyFeatures(X,p):

    X_poly = copy(X)


    for i in range(1,p):                                                     
        X_poly = c_[X_poly,X**(i+1)]                                         
                                                                             
                                                                             
    return X_poly                                                            
                                                                             
def featureNormalize(X_poly):                                                
                                                                             
    mu = mean(X_poly,axis=0)                                                 
                                                                             
    diff = X_poly - mu                                                       
                                                                             
    sigma = std(diff,axis=0,ddof =1)
                                                                             
    norm = diff/sigma                                                        
                                                                             
    return norm, mu, sigma                                                   
                                                                             





#########################

seed(1)

# LOAD CSV File
filename = 'sonar.all-data.csv'

dataset = list()
with open(filename, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		if not row:
			continue
		dataset.append(row)

# Separate features(X) from Class(Y)
for i in range(0, len(dataset[0])-1):
	for row in dataset:                            
		row[i] = float(row[i].strip())
 
X = [row[0:-1] for row in dataset] 
X = np.array(X)

Y_t = list()



for row in dataset:
	if row[-1] == 'R':
		Y_t.append(0)
	else:
		Y_t.append(1)

Y = np.array(Y_t)



# Split Training and Validation Set
p_Tr = 0.8
rnds = rand(len(Y))
iTr = find(rnds<p_Tr)
iVal = find(rnds>=p_Tr)
X_Tr = X[iTr,:]
Y_Tr = Y[iTr]
X_Val = X[iVal,:]
Y_Val = Y[iVal]


#Regularization
lam = 0.3


#Add and normalize polynomial features
#X_Tr = polyFeatures(X_Tr,2)
#X_Tr, mu, sigma = featureNormalize(X_Tr) 

#X_Val = polyFeatures(X_Val,2)
#X_Val, mu, sigma = featureNormalize(X_Val) 


#Concatenate 1's to X's
X_Tr = np.concatenate((np.ones((X_Tr.shape[0],1)),X_Tr),axis=1)
X_Val = np.concatenate((np.ones((X_Val.shape[0],1)),X_Val),axis=1)

#Validation Curve (Optimizing for Lambda) 
#lamda_vec       = array([0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]).T
#lam_err_tr, lam_err_val = validationCurve(X_Tr, Y_Tr, X_Val, Y_Val, lamda_vec)
#lam = lamda_vec[argmax(-lam_err_val)]
#print('Optimal Lambda = ', lam)


#Create Learning curve
#print('Running Learning Curve...')
#m_err_tr, m_err_val = learningCurve(X_Tr,Y_Tr,X_Val,Y_Val,lam)





#Train
theta = train(X_Tr,Y_Tr,lam)


#Predict on Training Set
pr = list()
for i in range(0,shape(X_Tr)[0]):
	pr.append(predict(theta, X_Tr[i]))

pr = np.array(pr)
err_Tr = len(find(pr != Y_Tr))/float(len(Y_Tr))
print('Error on Training Set', err_Tr)

#Predict on Validation Set
pr = list()
for i in range(0,shape(X_Val)[0]):
	pr.append(predict(theta, X_Val[i]))

pr = np.array(pr)
err_Val = len(find(pr != Y_Val))/float(len(Y_Val))
print('Error on Test Set', err_Val)





