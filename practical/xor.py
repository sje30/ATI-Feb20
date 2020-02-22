#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron
Original Source: https://github.com/Honghe/perceptron/tree/master/perceptron
Author: Honghe, modified by JVS, SJE, LDC.
"""

'''
The aim of this code is to demonstrate the incapacity of the perceptron to solve the XOR problem in two dimensions.
'''

import matplotlib.pyplot as plt
import numpy as np


# OPTIONAL: set random seed to get reproducible results
# np.random.seed(20)


class Perceptron(object):

    def __init__(self):
        super(Perceptron, self).__init__() #contructs the object instance
        self.w = np.array([np.random.rand() * 2 - 1 for _ in range(2)]) #initialise weights from an uniform distribution on [-1,1]
        self.learningRate = 0.001 #initialise learning rate
        self.maxiter = 100 #initialise maximal training iterations

    def output(self, x):  # perceptron output provided two dimensional data point x
        y = np.dot(x[range(2)],self.w) #dot product between data point x (we remove the label by selecting x[0,1]) and weights
        if y >= 0:
            res = 1.0
        else:
            res = -1.0
        return res #perceptron returns 1 if y is above decision boundary and -1 otherwise


    def updateWeights(self, x, iterError): #update weights given data point and misclassification error
        """
        updates the weights in the following way:
        w(t+1) = w(t) + learningRate * (d - r) * x
        iterError is (d - r)= desired output - actual output
        """
        self.w += self.learningRate * iterError * x[range(2)]


    def train(self, data):
        """
        Trains perceptron with all the data.
        Every vector in data must three elements.
        The third element of x (ie x[2]) is the label (desired output)
        """
        learned = False
        iteration = 0
        while not learned:
            numcorrect = 0    #number of correctly classified datapoints
            globalError = 0.0    #sum of iter errors over all datapoints
            for x in data:  # loop over datapoints
                r = self.output(x)
                if x[2] != r:  # if have a wrong response
                    iterError = x[2] - r  # desired response - actual response
                    self.updateWeights(x, iterError) #update weights to fit datapoint x
                    globalError += abs(iterError)    #
                else:
                    numcorrect += 1

            print('num correctly classified = %s' % numcorrect)
            iteration += 1
            if globalError == 0.0 or iteration >= self.maxiter:  # stop criteria
                print('iterations = %s' % iteration)
                learned = True  # stop learning

            ########## Plot ##########
            mis_classification_rate = 1-numcorrect/len(data)
            plotData(data, self.w, mis_classification_rate)          #function that plots data with decision boundary (specified later)
            plt.pause(0.05)                    #specifies how long the program stops to show the plot


def generateData(n):
    """
    generates data points corresponding to the XOR problem
    """
    data = np.array([0,0,0]) #starts data array with proxy datapoint
    for i in range(n):
        x = np.array([np.random.rand() * 2 - 1 for _ in range(2)])
        x = np.divide(x, np.abs(x)+1.e-17) #generates datapoint
        x += np.array([np.random.rand()/5 - 1/10 for _ in range(2)]) #adds some noise to the data
        x = np.append(x,-np.sign(np.prod(x))) #adds label corresponding to XOR problem
        data = np.vstack((data, x)) #adds datapoint to data array
    return data[1:,] #returns all datapoints except proxy datapoint


def plotData(data, w, r= -1):
    plt.clf()  # clear the figure
    for x in data: #loop over datapoints
        if x[2] > 0:
            plt.plot(x[0],x[1], 'ob')
        else:
            plt.plot(x[0],x[1], 'or')
    # plot the decision boundary.
    # The decision boundary is orthogonal to w.
    ww = np.divide(w,np.linalg.norm(w)+1.e-17)  # unit vector direction of w
    plt.ion()                       #turn the interactive mode on
    plt.plot([ww[1],-ww[1]],[-ww[0],ww[0]],'--k') #draws the orthogonal decision boundary
    plt.suptitle('Two classes separated by a line orthogonal to the weight vector')
    if r != -1:
        plt.title('Misclassification rate. = %s' % r)
    plt.show()


##ACTUAL CODE:

#number of datapoints
n=10

# generate data
trainset = generateData(n)  # train set generation

# declare perceptron
p = Perceptron()

# plot data and decision boundary
plotData(trainset, p.w)

# train perceptron
p.train(trainset)

# plot data and decision boundary
plotData(trainset, p.w)

########## The End ##########