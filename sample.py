'''
https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html
as a working example I first try this example
'''

from Classes.image_matching import ImageMatching
from Classes.openCVHDRTutorial import HDRTutorial

if __name__ == '__main__':
    image_matching = ImageMatching()
    image_matching.load_images("Images/Wikipedia/")

    # by replacing to light or to dark values with random values i try to stop sift to be interested in this parts
    image_matching.radomize_unwanted()

    image_matching.display()
    image_matching.create_sift()


    sample = HDRTutorial()
    path = "Images/Wikipedia/"
    sample.create_hdr(path)
    sample.display()

