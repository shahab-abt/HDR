import cv2 as cv
import numpy as np

class WarpPerspective:
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image1 = image2

    def calculate_warp_perspective(self, image1, image2):
        sift = cv.SIFT_create()
        key_points_1, descriptors_1 = sift.detectAndCompute(image1, None)
        key_points_2, descriptors_2 = sift.detectAndCompute(image2, None)
        MIN_MATCH_COUNT = 10


        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([key_points_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([key_points_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            h, w = image1.shape

            out = cv.warpPerspective(image1, M, (w, h), flags=cv.INTER_LINEAR)
            return out
        else:
            return False
