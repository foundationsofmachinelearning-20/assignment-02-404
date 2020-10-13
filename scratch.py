import math
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



# functions

## Feature Extraction for image that we have to read in

# EdgeHistogramComputer, modified from public feature extraction code found on
# https://github.com/scferrada/imgpedia

class EdgeHistogramComputer:

    def __init__(self, rows, cols):
        sqrt2 = math.sqrt(2)
        self.kernels = (np.array([[1, 1], [-1, -1]]), #Vertical Edge
                        np.array([[1, -1], [1, -1]]), # Horizontal edge
                        np.array([[sqrt2, 0], [0, -sqrt2]]), # Diagonal (45)
                        np.array([[0, sqrt2], [-sqrt2, 0]]), # diagaonal (135)
                        np.array([[2, -2], [-2, 2]])) #  Non-Orientation
        self.bins = [len(self.kernels)]
        self.range = [0, len(self.kernels)]
        self.rows = rows
        self.cols = cols
        self.prefix = "EDH"

    def compute(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        descriptor = []
        dominantGradients = np.zeros_like(frame)
        maxGradient = cv2.filter2D(frame, cv2.CV_32F, self.kernels[0])
        maxGradient = np.absolute(maxGradient)
        for k in range(1, len(self.kernels)):
            kernel = self.kernels[k]
            gradient = cv2.filter2D(frame, cv2.CV_32F, kernel)
            gradient = np.absolute(gradient)
            np.maximum(maxGradient, gradient, maxGradient)
            indices = (maxGradient == gradient)
            dominantGradients[indices] = k

        frameH, frameW = frame.shape
        for row in range(self.rows):
            for col in range(self.cols):
                mask = np.zeros_like(frame)
                mask[int(((frameH / self.rows) * row)):int(((frameH / self.rows) * (row + 1))),
                int((frameW / self.cols) * col):int(((frameW / self.cols) * (col + 1)))] = 255
                hist = cv2.calcHist([dominantGradients], [0], mask, self.bins, self.range)
                hist = cv2.normalize(hist, None)
                descriptor.append(hist)

        # return np.concatenate([x for x in descriptor])
        descriptor = np.array(descriptor)
        globalEdges = np.transpose(descriptor.mean(0))[...,None]
        descriptor = np.append(descriptor, globalEdges ,axis=0)
        descriptor = np.squeeze(descriptor,2)
        return descriptor



#Load testing data and information for trained model
#test points
Features_test = np.load('Features_test.npy')
Labels_test=np.load('Labels_test.npy')
#trained values
[[ class0_COV],
       [ class1_COV],
       [ class2_COV],
       [ class3_COV],
       [ class4_COV]]=np.load('Trained_COV.npy')
[[class0_MUS],
    [class1_MUS],
    [class2_MUS],
    [class3_MUS],
    [class4_MUS]]=np.load('Trained_MU.npy')
#print('Class 0 Mus',[class0_MUS])
#print('Class 2 COV matrix',[class2_COV])


# testing
posc0 = multivariate_normal.pdf(Features_test, class0_MUS, class0_COV, True)
posc1 = multivariate_normal.pdf(Features_test, class1_MUS, class1_COV, True)
posc2 = multivariate_normal.pdf(Features_test, class2_MUS, class2_COV, True)
posc3 = multivariate_normal.pdf(Features_test, class3_MUS, class3_COV, True)
posc4 = multivariate_normal.pdf(Features_test, class4_MUS, class4_COV, True)


#build confusion Matrix
prediction_temp = 0
predictedlabels = []


for i in range(posc0.size):
    temp_vect = np.array([posc0[i], posc1[i], posc2[i], posc3[i], posc4[i]])
    prediction_temp = np.argmax(temp_vect)
    predictedlabels.append(prediction_temp)

predictedlabels = np.array(predictedlabels)
"""
#use to check %correct
correct = 0
total = 0

for k in range(predictedlabels.size):
    if predictedlabels[k] == Labels_test[k]:
        correct += 1
    total += 1
print(correct/total)
"""
confusionMatrix = np.zeros((5,5))

for i in range(Labels_test.size):
    confusionMatrix[Labels_test[i],[predictedlabels[i]]] += 1


print("Confusion Matrix:")
print(confusionMatrix)