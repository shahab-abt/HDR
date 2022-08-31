'''
https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html
as a working example I first try this example
'''

import cv2 as cv
import numpy as np

from Images.Classes.openCVHDRTutorial import HDRTutorial

if __name__ == '__main__':

    sample = HDRTutorial()
    path = "Images/Wikipedia/"
    sample.create_hdr(path)

