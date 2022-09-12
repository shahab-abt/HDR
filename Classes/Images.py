import cv2 as cv
from os import listdir
import numpy as np
from os.path import isfile, join


class Images:
    def __init__(self, path):
        self.images = None
        self.path = path
        self.load_images()
        self.sort_images()

    def load_images(self):
        img_fn = [self.path + f for f in listdir(self.path) if isfile(join(self.path, f))]
        self.images = [cv.imread(fn) for fn in img_fn]
        self.images = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in self.images]

    def sort_images(self):
        average = self.get_average()
        unsorted_img = self.images
        # Z = self.images.sort(key=average.tolist() )

        sortedIm = [unsorted_img for _, unsorted_img in sorted(zip(average, unsorted_img))]
        sorted_img =[]

        a = sorted(zip(average,unsorted_img),reverse=True)
        for img in a:
            print(img[0])
            print(np.average(img[1]))
            sorted_img.append(img[1])


        self.images = sorted_img

        for i in range(len(sorted_img)):
            cv.imwrite("raw" + str(i) + ".jpg", sorted_img[i])
            print("raw" + str(i) + " avr: " + str(np.average(sorted_img[i])))
        '''        
        i = 0
        for img in sorted_img:
            cv.imwrite("raw" + str(i) + ".jpg", img)
            print("raw" + str(i) +" avr: "+str(np.average(img)))
            i += 1
        '''

    def get_average(self):
        average = [np.average(img) for img in self.images]
        # img = self.images[index]
        # print(average)
        # print(np.sort(average))
        #print(np.sort(average)[::-1])
        # average = average.sort(reverse=True)
        # return np.average(img)
        #return np.sort(average)[::-1]
        return average
