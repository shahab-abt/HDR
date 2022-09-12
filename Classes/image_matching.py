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
        self.keypoints_1 =None
        self.descriptors_1 =None
        self.keypoints_2 = None
        self.descriptors_2 = None
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
        self.keypoints_1, self.descriptors_1 = self.sift.detectAndCompute(self.images[0], None)
        self.keypoints_2, self.descriptors_2 = self.sift.detectAndCompute(self.images[1], None)



    #def find_give_good_matches(self):
    def find_Homography(self):
        '''
        source link:
        https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
        '''
        MIN_MATCH_COUNT = 10
        MAX_MATCH_COUNT = 50

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.descriptors_1, self.descriptors_2, k=2)
        good_matches = []



        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                #if len(good_matches) > MAX_MATCH_COUNT:
                #    break
        #return good_matches
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            '''
            temp =good_matches.copy()
            for m in good_matches:
                print(self.keypoints_1[m.trainIdx].pt)

            for i in good_matches.shape[0]:
                m = good_matches[i]
                print(self.keypoints_1[m.trainIdx].pt)
            '''
            dst_pts = np.float32([self.keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            mask_reshaped = np.array( mask.reshape(mask.shape[0]),dtype=bool)
            src_pts_masked = src_pts[mask_reshaped,:,:]
            dst_pts_masked = dst_pts[mask_reshaped,:,:]
            #slected = self.get_neighbor_match(1000, 1000, src_pts_masked, dst_pts_masked)
            matchesMask = mask.ravel().tolist()
            h, w = self.images[0].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            out = cv.warpPerspective(self.images[0], M, (w, h), flags=cv.INTER_LINEAR)
            cv.imwrite("imgwarp.jpg", out)
            img2 = cv.polylines(self.images[1], [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            cv.imwrite("img2.jpg", img2)
            cv.imwrite("img1.jpg", self.images[0])
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            matchesMask = None
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(self.images[0], self.keypoints_1, img2, self.keypoints_2, good_matches, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()
        cv.imwrite("newCompare.jpg", img3)
        print("done")

    def get_neighbor_match(self, u, v,src_pts, dst_pts ):
        u_range = 50
        v_range = 50
        point =np.array([[u,v]])
        src = src_pts[:, 0, :]
        dst = dst_pts[:, 0, :]

        #src = np.vstack([src,np.array([0,u])])
        #src = np.vstack([src, np.array([v, 0])])
        dist = np.linalg.norm((src - point), axis=1).reshape(src.shape[0],1)
        src =np.concatenate((src,dist),axis=1)
        src = np.concatenate((src, dst), axis=1)
        src = np.unique(src, axis=0)
        src = src[src[:, 2].argsort()]
        selected = src[:4,:]


        src_p = np.array(selected[:,0:2],np.float32)
        dst_p = np.array(selected[:,3:],np.float32)
        #dtype=float32)

        retH, maskH = cv.findHomography(src_p.reshape(-1, 1, 2) ,dst_p.reshape(-1, 1, 2), cv.RANSAC, 5.0)

        selected = selected[selected[:, 3].argsort()]
        src_p = np.array(selected[:,0:2],np.float32)
        dst_p = np.array(selected[:,3:],np.float32)
        ret2 = cv.getPerspectiveTransform( src_p,dst_p)

        pixel = np.array([[u,v,1],[u+1,v,1],[u,v+1,1],[u+1,v+1,1]])
        result = np.matmul(pixel,ret.transpose())
        onC = np.array([[1],[1],[1],[1]])


        dsD = np.matmul( np.concatenate((dst_p,onC),axis=1) ,ret.transpose())
        dsS = np.matmul(np.concatenate((src_p,onC),axis=1) , ret.transpose())


        return selected

        #test Block
        src = np.float32([[0, 0], [0, 3455], [5183, 3455], [5183, 0]])
        dts = np.float32([[25.77396, -19.07275], [19.25586, 3435.28687], [5203.10889, 3444.92603], [5210.27197, -12.18482]])
        M = cv.getPerspectiveTransform(src, dts)
        onC = np.array([[1], [1], [1], [1]])
        control = np.matmul( np.concatenate((src,onC),axis=1) ,M.transpose())

        src = src_pts[:,0,:]
        dst = dst_pts[:,0,:]
        src = src[src[:, 0].argsort()]
        selected = src.copy()
        selected = selected[selected[:,0] > (u -u_range) ]
        selected = selected[selected[:, 0] < (u + u_range)]
        selected = selected[selected[:, 1].argsort()]
        selected = selected[selected[:,1] > (v -v_range) ]
        selected = selected[selected[:, 1] < (v+ v_range)]
        print( selected)

        pass


'''
a,b  = len(self.keypoints_1), len(self.keypoints_2)

# feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = bf.match(self.descriptors_1, self.descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv.drawMatches(self.images[0], self.keypoints_1, self.images[1], self.keypoints_2, matches[:50], self.images[1], flags=2)
plt.imshow(img3)
cv.imwrite( "Tes.jpg", img3)
plt.show()
'''


