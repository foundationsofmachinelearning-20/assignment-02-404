"""
File:   Assignment2.py
Author: Rachel Singleton
Due date: Oct 8, 2020
   
"""


#req libraries
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.feature
import skimage.viewer
import sys
import scipy.stats as stats
""" =======================  Image processing functions from the internet: ======================= """
#
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def pic_edges(image):
    # https://datacarpentry.org/image-processing/08-edge-detection/
    sigma = 1.5 #2 works best
    low_threshold = 0.4 #between 0 and 1
    high_threshold = 1 #between 0 and 1


    temp=rgb2gray(image)
    edges = skimage.feature.canny(
        image=temp,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    # display edges
    #viewer = skimage.viewer.ImageViewer(edges)
    #viewer.show()
    return edges
""" =======================  Built functions ======================= """
def count_length(edge_temp): 
    #takes in matrix that has already run through edge detection
    #counts the number of bricks
    #finds all lengths of bricks in horizontal direction
    temp=0
    brick_count=0
    length=[]
    height=[]
    #should move accross row (so we need number of cols in row)
    for j in range(np.size(edge_temp, 1)):
        #This should move accross cols (so we need number of rows in col)
        for i in range(np.size(edge_temp, 0)):
            if edge_temp[j,i]: #true is edge, false is not
                if temp>=5:#if it is smaller than this, we assume edge detector was noisy
                    if temp<25:
                        #This is an ok assumption based off what we know about input pictures
                        length+=[temp]
                        temp=0
                        brick_count+=1
                    else: #if the length is really big (1/4 the picture), just ignore it bc that was probably 
                        #something else in the picture or a bad edge reading
                        temp=0

            else:
                #find length of this brick
                temp+=1
                #rint(length_temp)
        #Move in opposite pattern to find heights
    temp=0
    for j in range(np.size(edge_temp,0)):     
        for i in range(np.size(edge_temp, 1)):
            if edge_temp[i,j]: #true is edge, false is not
                if temp>=2:#if it is smaller than this, we assume edge detector was noisy
                    if temp<9:#based off of looking at print out, really consistantly under 5
                        #This is an ok assumption based off what we know about input pictures
                        height+=[temp]
                        temp=0
                    else: #if the length is really big (1/4 the picture), just ignore it bc that was probably 
                        #something else in the picture or a bad edge reading
                        temp=0

            else:
                #find length of this brick
                temp+=1
                #rint(length_temp)
    
    brick_count=brick_count/np.average(height)
  #print('Brick count= ', brick_count)
   #print(length)
    return brick_count, length, height



""" =======================  Load Data ======================= """
img = np.load('images.npy', allow_pickle=True)
Labels= np.load('Labels.npy', allow_pickle=True)
""" ======================= Seperate into training and Testing ======================= """
#Take data from each group at random.... 
#This dataset is only 121 points
train_d=[]
train_L=[]
test_d=[]
test_L=[]
for i in range(np.size(Labels)): #0-121 for this data set
    #About 1/5 will be training and 4/5 will be testing
    if i%5==0: #if divisable by 5
        train_d+=[img[i]]
        train_L+=[Labels[i]]
    else:
        test_d+=[img[i]]
        test_L+=[Labels[i]]

#print(np.size(train_L))
#print(np.size(test_L))
""" ========================  Extract Features ============================== """
####                | average heights of corresponsding image |
####                | average lengths of corresponding image  |
#### feature matrix=|   calculated brick count of image       |
####                |              truth data                 |
""" Training """
#Extract average horizontal length, vertical height and brick count
train_feat=np.zeros((4, np.size(train_L)))

for k in range(np.size(train_L)):
    edge_temp=pic_edges(train_d[k])
    [brick_count, length, height]=count_length(edge_temp)
    train_feat[0,k]=np.average(height)
    train_feat[1,k]=np.average(length)
    train_feat[2,k]=brick_count
    train_feat[3,k]=train_L[k]


""" Testing """
#Extract average horizontal length, vertical height and brick count
test_feat=np.zeros((4, np.size(test_L)))

for k in range(np.size(test_L)):
    edge_temp=pic_edges(test_d[k])
    [brick_count, length, height]=count_length(edge_temp)
    test_feat[0,k]=np.average(height)
    test_feat[1,k]=np.average(length)
    test_feat[2,k]=brick_count
    test_feat[3,k]=test_L[k]


""" ========================  Train Data ============================== """
""" Probabilistic Generative Classifier """
#find our mu's and sigma's
C1_h=[]
C1_l=[]
C1_b=[]

C2_h=[]
C2_l=[]
C2_b=[]

C3_h=[]
C3_l=[]
C3_b=[]

C4_h=[]
C4_l=[]
C4_b=[]

for k in range(np.size(train_L)):
    if train_feat[3,k]==1:
        C1_h+=[train_feat[0,k]]
        C1_l+=[train_feat[1,k]]
        C1_b+=[train_feat[2,k]]
    elif train_feat[3,k]==2:
        C2_h+=[train_feat[0,k]]
        C2_l+=[train_feat[1,k]]
        C2_b+=[train_feat[2,k]]
    elif train_feat[3,k]==3:
        C3_h+=[train_feat[0,k]]
        C3_l+=[train_feat[1,k]]
        C3_b+=[train_feat[2,k]]
    elif train_feat[3,k]==4:
        C4_h+=[train_feat[0,k]]
        C4_l+=[train_feat[1,k]]
        C4_b+=[train_feat[2,k]] 

C1_mu_H=np.average(C1_h)
C1_mu_L=np.average(C1_l)
C1_mu_B=np.average(C1_b)

C2_mu_H=np.average(C2_h)
C2_mu_L=np.average(C2_l)
C2_mu_B=np.average(C2_b)

C3_mu_H=np.average(C3_h)
C3_mu_L=np.average(C3_l)
C3_mu_B=np.average(C3_b)

C4_mu_H=np.average(C4_h)
C4_mu_L=np.average(C4_l)
C4_mu_B=np.average(C4_b)


C1_var_h=0
C1_var_l=0
C1_var_b=0
#could make this a function, but I'm not going to bc this will run faster :)
for k in range(np.size(C1_h)): #should all be the same length within a class
    C1_var_h+=(C1_h[k]-C1_mu_H)**2
    C1_var_l+=(C1_l[k]-C1_mu_L)**2
    C1_var_b+=(C1_b[k]-C1_mu_B)**2

C1_var_h=np.sqrt(C1_var_h/np.size(C1_h))
C1_var_l=np.sqrt(C1_var_l/np.size(C1_h))
C1_var_b=np.sqrt(C1_var_b/np.size(C1_h))




    
C2_var_h=0
C2_var_l=0
C2_var_b=0

for k in range(np.size(C2_h)): #should all be the same length within a class
    C2_var_h+=(C2_h[k]-C2_mu_H)**2
    C2_var_l+=(C2_l[k]-C2_mu_L)**2
    C2_var_b+=(C2_b[k]-C2_mu_B)**2

C2_var_h=np.sqrt(C2_var_h/np.size(C2_h))
C2_var_l=np.sqrt(C2_var_l/np.size(C2_h))
C2_var_b=np.sqrt(C2_var_b/np.size(C2_h))



C3_var_h=0
C3_var_l=0
C3_var_b=0

#could make this a function, but I'm not going to bc this will run faster :)
for k in range(np.size(C3_h)): #should all be the same length within a class
    C3_var_h+=(C3_h[k]-C3_mu_H)**2
    C3_var_l+=(C3_l[k]-C3_mu_L)**2
    C3_var_b+=(C3_b[k]-C3_mu_B)**2

C3_var_h=np.sqrt(C3_var_h/np.size(C3_h))
C3_var_l=np.sqrt(C3_var_l/np.size(C3_h))
C3_var_b=np.sqrt(C3_var_b/np.size(C3_h))



C4_var_h=0
C4_var_l=0
C4_var_b=0
#could make this a function, but I'm not going to bc this will run faster :)
for k in range(np.size(C4_h)): #should all be the same length within a class
    C4_var_h+=(C4_h[k]-C4_mu_H)**2
    C4_var_l+=(C4_l[k]-C4_mu_L)**2
    C4_var_b+=(C4_b[k]-C4_mu_B)**2

C4_var_h=np.sqrt(C4_var_h/np.size(C4_h))
C4_var_l=np.sqrt(C4_var_l/np.size(C4_h))
C4_var_b=np.sqrt(C4_var_b/np.size(C4_h))



""" ========================  Test Data ============================== """
""" Probabilistic Generative Classifier """

""" ========================  Print Confusion Matricies ============================== """
