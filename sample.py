'''
https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html
as a working example I first try this example
'''
import cv2 as cv
import numpy as np
from Classes.Images import Images
from Classes.WarpPerspective import WarpPerspective
from Classes.image_matching import ImageMatching
from Classes.openCVHDRTutorial import HDRTutorial

if __name__ == '__main__':
    images = Images("Images/images2")
    warping = WarpPerspective(images.images[0], images.images[1])
    source_img = images.images[0]
    calculated=[]
    #calculated.append(images.images[0])
    #cv.imwrite("Exporting/img"+str(0)+".jpg", source_img)
    source_img = images.images[0]
    calculated.append(source_img)
    for i in range(len(images.images)-1):

        #source_img = warping.calculate_warp_perspective(source_img,images.images[i])
        source_img = warping.calculate_warp_perspective( images.images[i+1],source_img)
        calculated.append(source_img)
        print(np.average( images.images[i+1]))

    for i in range(len(calculated)):

        cv.imwrite("Exporting/img"+str(i)+".jpg", calculated[i])

    image_matching = ImageMatching()
    image_matching.load_images("Images/Wikipedia/")

    # by replacing to light or to dark values with random values i try to stop sift to be interested in this parts
    #image_matching.radomize_unwanted()

    #image_matching.display()
    image_matching.create_sift()
    image_matching.find_Homography()


    sample = HDRTutorial()
    path = "Images/Wikipedia/"
    sample.create_hdr(path)
    sample.display()

