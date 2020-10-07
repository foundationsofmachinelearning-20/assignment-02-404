# relavent import Statements

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import math
from FeatureExtractors.EdgeHistogramComputer import EdgeHistogramComputer


# functions

def featureMUandVar(Features, Labels, desiredclass):
    # concatenate the feature array and the Labels array
    Features_and_Labels = np.hstack((Features, Labels[..., None]))
    # mask all but the desired class
    possibleClasses = [0, 1, 2, 3, 4]
    possibleClasses.remove(desiredclass)
    for classes in possibleClasses:
        Features_and_Labels = np.ma.masked_equal(Features_and_Labels, classes)
    Features_and_Labels = np.ma.mask_rows(Features_and_Labels)
    Features_and_Labels = [m.compressed() for m in Features_and_Labels]
    Features_and_Labels = np.array([i for i in Features_and_Labels if 0 not in i.shape])
    Features_and_Labels = np.delete(Features_and_Labels, 5, 1)
    MUS = Features_and_Labels.mean(0)
    COV = np.ma.cov(np.transpose(Features_and_Labels))
    pc = Features_and_Labels.shape[0] / (Features.shape[0])
    return MUS, COV, Features_and_Labels, pc


# Feature Extraction
Images = np.load("Images.npy")
Labels = np.load("Labels.npy")
EHDComp = EdgeHistogramComputer(4, 4)

# Features is a 107 x 5 matrix, where the row corresponds to the image, col 1 is VE, col2 is HE, col3, is D45
# col 4 is D135, and col5 is NO. The data points are averages taken accross the entire image
Features = []

for data in Images:
    IMG_EHD = EHDComp.compute(data)
    Features.append(IMG_EHD[16])
Features = np.array(Features)

plt.figure(1)
plt.bar([1, 2, 3, 4, 5], Features[16])
plt.xticks([1, 2, 3, 4, 5], ['VE', 'HE', 'D45', 'D135', 'NO'])
plt.show()

# Finding Mus, Variances, and priors
# There are 5 possible classes, and 5 features extracted from each image, therefore each feature will need a MU
# Variance for each feature, ie 50 variables :/. To not have 50 variables, Im just going to use lists for each class.
# classN_MUS = [VE_MU, HE_MU, D45_MU, D135_MU, NO_MU], and same format for variances.


# class 0
class0_MUS, class0_COV, class0_Features, pc0 = featureMUandVar(Features, Labels, 0)
print(np.amax(class0_COV))

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

plt.show()

# visualizing with multivariate norm


x1 = np.linspace(-1, 1, 10)
x2 = np.linspace(-1,1,10) #commented out, used for other features if nessecary
x3 = np.linspace(-1,1,10)
x4 = np.linspace(-1,1,10)
x5 = np.linspace(-1,1,10)

x1m, x2m, x3m, x4m, x5m = np.meshgrid(x1, x2, x3, x4, x5)
X = np.stack([x1m, x2m, x3m, x4m, x5m], -1)

y0 = multivariate_normal.pdf(X, mean=class0_MUS, cov=class0_COV,allow_singular=True)
y1 = multivariate_normal.pdf(X, mean=class1_MUS, cov=class1_COV)
y2 = multivariate_normal.pdf(X, mean=class2_MUS, cov=class2_COV)
y3 = multivariate_normal.pdf(X, mean=class3_MUS, cov=class3_COV)
y4 = multivariate_normal.pdf(X, mean=class4_MUS, cov=class4_COV,allow_singular=True)


# calculate posteriors
pos0 = (y0*pc0)/(y0*pc0 + y1*pc1 + y2*pc2 + y3*pc3 + y4*pc4)
pos1 = (y1*pc1)/(y0*pc0 + y1*pc1 + y2*pc2 + y3*pc3 + y4*pc4)
pos2 = (y2*pc2)/(y0*pc0 + y1*pc1 + y2*pc2 + y3*pc3 + y4*pc4)
pos3 = (y3*pc3)/(y0*pc0 + y1*pc1 + y2*pc2 + y3*pc3 + y4*pc4)
pos4 = (y4*pc4)/(y0*pc0 + y1*pc1 + y2*pc2 + y3*pc3 + y4*pc4)




