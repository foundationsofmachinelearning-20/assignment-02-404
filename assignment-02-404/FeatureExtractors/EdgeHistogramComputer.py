import cv2
import numpy as np
import math


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
