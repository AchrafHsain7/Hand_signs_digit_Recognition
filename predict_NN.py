import numpy as np


def reLU( Z):
        A = np.maximum(0, Z)
        
        return A
def softMax(Z):
    e_z = np.exp(Z - np.max(Z))
    return e_z / e_z.sum()


#forward propagation
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    Z1 = Z1.reshape(25,1)
    A1 = reLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    Z2 = Z2.reshape(12,1)
    A2 = reLU(Z2)
    Z3 = np.dot(W3, A2) + b3
    Z3 = Z3.reshape(6,1)
    #print(Z1.shape)
    #print(Z2.shape)
    #print(Z3.shape)

    return Z3

#predicting
def predict(X, parameters):
    Z3 = forward_propagation(X, parameters)
    A3 = softMax(Z3)

    return A3