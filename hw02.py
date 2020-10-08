import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats as stats
import math


# functions

#EdgeHistogramComputer, modified from public feature extraction code found on
#https://github.com/scferrada/imgpedia

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


def featureMUandVar(Features, Labels, desiredclass):
    # concatenate the feature array and the Labels array
    Features_and_Labels = np.hstack((Features, Labels[..., None]))
    # mask all but the desired class and create matrix of only the class
    possibleClasses = [0, 1, 2, 3, 4]
    possibleClasses.remove(desiredclass)
    for classes in possibleClasses:
        Features_and_Labels = np.ma.masked_equal(Features_and_Labels, classes)
    Features_and_Labels = np.ma.mask_rows(Features_and_Labels)
    Features_and_Labels = [m.compressed() for m in Features_and_Labels]
    Features_and_Labels = np.array([i for i in Features_and_Labels if 0 not in i.shape])
    Features_and_Labels = np.delete(Features_and_Labels, 5, 1)
    # calculate nessecary parameters
    MUS = Features_and_Labels.mean(0)
    COV = np.ma.cov(np.transpose(Features_and_Labels))
    pc = Features_and_Labels.shape[0] / (Features.shape[0])
    return MUS, COV, Features_and_Labels, pc


# Feature Extraction
Images = np.load("Images.npy")
Labels = np.load("Labels.npy")
Images_2 = np.load("Images_2.npy")
Labels_2 = np.load("Labels_2.npy")
EHDComp = EdgeHistogramComputer(4, 4)

# Features is a row x 5 matrix, where the row corresponds to the image, col 1 is VE, col2 is HE, col3, is D45
# col 4 is D135, and col5 is NO. The data points are averages taken accross the entire image
Features_temp = []

for data in Images:
    IMG_EHD = EHDComp.compute(data)
    Features_temp.append(IMG_EHD[16])

for data in Images_2:
    IMG_EHD = EHDComp.compute(data)
    Features_temp.append(IMG_EHD[16])

Features_temp = np.array(Features_temp)
Labels_temp = np.append(Labels, Labels_2)

Features_test = []
Labels_test = []
Features = []
Labels = []

# split data into testing data and training data

for i in range(np.size(Labels_temp)):  # 0-121 for this data set
    # About 1/5 will be training and 4/5 will be testing
    if i % 3 == 0 and Labels_temp[i] != 0:  # if divisable by 5
        Features_test.append(Features_temp[i])
        Labels_test.append(Labels_temp[i])
    else:
        Features.append(Features_temp[i])
        Labels.append(Labels_temp[i])

Features = np.array(Features)
Labels = np.array(Labels)
Features_test = np.array(Features_test)
Labels_test = np.array(Labels_test)
print("Features Vector of One Image: ")
print(Features[1])


# plt.figure(1)
# plt.bar([1, 2, 3, 4, 5], Features[16])
# plt.xticks([1, 2, 3, 4, 5], ['VE', 'HE', 'D45', 'D135', 'NO'])
# plt.show()

# Finding Mus, Variances, and priors
# There are 5 possible classes, and 5 features extracted from each image, therefore each feature will need a MU
# Variance for each feature, ie 50 variables :/. To not have 50 variables, Im just going to use lists for each class.
# classN_MUS = [VE_MU, HE_MU, D45_MU, D135_MU, NO_MU], and same format for variances.


# class 0
class0_MUS, class0_COV, class0_Features, pc0 = featureMUandVar(Features, Labels, 0)
# class 1
class1_MUS, class1_COV, class1_Features, pc1 = featureMUandVar(Features, Labels, 1)
# class 2
class2_MUS, class2_COV, class2_Features, pc2 = featureMUandVar(Features, Labels, 2)
# class 3
class3_MUS, class3_COV, class3_Features, pc3 = featureMUandVar(Features, Labels, 3)
# class 4
class4_MUS, class4_COV, class4_Features, pc4 = featureMUandVar(Features, Labels, 4)
# Plotting the features for graphs

plt.figure(1)

plt.plot(class1_Features[:, 0], class1_Features[:, 1], 'bo', label=' class 1 features 1 v 2')
plt.plot(class2_Features[:, 0], class2_Features[:, 1], 'go', label='class 2 features 1 v 2')
plt.plot(class3_Features[:, 0], class3_Features[:, 1], 'ro', label='class 3 featires 1 v 2')

plt.figure(2)
plt.hist(class1_Features[:, 2], 50)
# plt.show()

# visualizing with multivariate norm


x1 = np.linspace(-1, 1, 10)
x2 = np.linspace(-1, 1, 10)  # commented out, used for other features if nessecary
x3 = np.linspace(-1, 1, 10)
x4 = np.linspace(-1, 1, 10)
x5 = np.linspace(-1, 1, 10)

x1m, x2m, x3m, x4m, x5m = np.meshgrid(x1, x2, x3, x4, x5)
X = np.stack([x1m, x2m, x3m, x4m, x5m], -1)

# generating probability functions

y0 = multivariate_normal.pdf(X, mean=class0_MUS, cov=class0_COV, allow_singular=True)
y1 = multivariate_normal.pdf(X, mean=class1_MUS, cov=class1_COV)
y2 = multivariate_normal.pdf(X, mean=class2_MUS, cov=class2_COV)
y3 = multivariate_normal.pdf(X, mean=class3_MUS, cov=class3_COV)
y4 = multivariate_normal.pdf(X, mean=class4_MUS, cov=class4_COV, allow_singular=True)

# calculate posteriors
#pos0 = (y0 * pc0) / (y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4)
#pos1 = (y1 * pc1) / (y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4)
#pos2 = (y2 * pc2) / (y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4)
#pos3 = (y3 * pc3) / (y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4)
#pos4 = (y4 * pc4) / (y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4)

# Testing Data
#Images_test = np.load("image.npy")
#Labels_test = np.load("label.npy")

# Images_test = np.delete(Images_test,104,0)

#print(Images_test.shape)
#print(Labels_test.shape)

#Features_test = []

#for data in Images_test:
#    IMG_EHD = EHDComp.compute(data)
#    Features_test.append(IMG_EHD[16])
#Features_test = np.array(Features_test)

#print(Features_test)

# testing
posc0 = multivariate_normal.pdf(Features_test, class0_MUS, class0_COV, True)
posc1 = multivariate_normal.pdf(Features_test, class1_MUS, class1_COV, True)
posc2 = multivariate_normal.pdf(Features_test, class2_MUS, class2_COV, True)
posc3 = multivariate_normal.pdf(Features_test, class3_MUS, class3_COV, True)
posc4 = multivariate_normal.pdf(Features_test, class4_MUS, class4_COV, True)

prediction_temp = 0
predictedlabels = []


for i in range(posc0.size):
    temp_vect = np.array([posc0[i], posc1[i], posc2[i], posc3[i], posc4[i]])
    prediction_temp = np.argmax(temp_vect)
    predictedlabels.append(prediction_temp)

predictedlabels = np.array(predictedlabels)

correct = 0
total = 0

for k in range(predictedlabels.size):
    if predictedlabels[k] == Labels_test[k]:
        correct += 1
    total += 1
#print(correct/total)
confusionMatrix = np.zeros((5,5))

check_c2_ca_c1 = 0




for i in range(Labels_test.size):
    confusionMatrix[Labels_test[i],[predictedlabels[i]]] += 1


print("Confusion Matrix:")
print(confusionMatrix)