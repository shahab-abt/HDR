import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt
import random

'''
source tutorial
https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
'''

class ImageMatching:
    def __init__(self):
        self.images =None
        self.sift = None

    # TODO findHomography


    def load_images(self,path):
        img_fn = [path + f for f in listdir(path) if isfile(join(path, f))]
        self.images = [cv.imread(fn) for fn in img_fn]
        self.images = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in self.images ]


    def radomize_unwanted(self):
        # TODO should create a mask layer from unwanted values and replace all unwanted based on mask with a random numpy

        for k in range (2):
            width = self.images[k].shape[0]
            lenght = self.images[k].shape[1]
            for i in range (width):
                for j in range (lenght):
                    if(self.images[k][i,j]>250 or self.images[k][i,j]<5):
                        self.images[k][i, j] = random.randint(0, 255)

        '''
        pic = self.images[i]
        pic[pic>250] = random.randint(0, 255)
        self.images[i] = pic

        pic = self.images[i]
        pic[pic < 5] = random.randint(0, 255)
        self.images[i] = pic
        

        for k in range(2):
            pic = self.images[k]
            pic[pic > 210] = 0
            self.images[k] = pic

            pic = self.images[k]
            pic[pic < 40] = 0
            self.images[k] = pic
        '''



    def display(self):
        figure, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(self.images[0], cmap='gray')
        ax[1].imshow(self.images[1], cmap='gray')
        plt.show()


    def create_sift(self):
        self.sift = cv.SIFT_create()
        keypoints_1, descriptors_1 = self.sift.detectAndCompute(self.images[0], None)
        keypoints_2, descriptors_2 = self.sift.detectAndCompute(self.images[1], None)
        a,b  = len(keypoints_1), len(keypoints_2)

        # feature matching
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv.drawMatches(self.images[0], keypoints_1, self.images[1], keypoints_2, matches[:50], self.images[1], flags=2)
        plt.imshow(img3)
        cv.imwrite( "Tes.jpg", img3)
        plt.show()


