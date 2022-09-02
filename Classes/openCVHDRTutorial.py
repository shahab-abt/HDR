'''
https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html
create HDR photod with openCV tutorial
'''

import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from pathlib import Path



class HDRTutorial:

    def create_hdr(self,path):
        img_fn = [path+f for f in listdir(path) if isfile(join(path, f))]
        img_list = [cv.imread(fn) for fn in img_fn]
        exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

        # Merge exposures to HDR image
        merge_debevec = cv.createMergeDebevec()
        hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
        merge_robertson = cv.createMergeRobertson()
        hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

        # Tonemap HDR image
        tonemap1 = cv.createTonemap(gamma=2.2)
        res_debevec = tonemap1.process(hdr_debevec.copy())

        # Exposure fusion using Mertens
        merge_mertens = cv.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)

        # Convert datatype to 8-bit and save
        res_debevec_8bit = np.clip(res_debevec * 255, 0, 255).astype('uint8')
        res_robertson_8bit = np.clip(hdr_robertson * 255, 0, 255).astype('uint8')
        res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')

        export_path = path+"export"
        Path(export_path).mkdir(parents=True, exist_ok=True)

        cv.imwrite(export_path+"/ldr_debevec.jpg", res_debevec_8bit)
        cv.imwrite(export_path+"/ldr_robertson.jpg", res_robertson_8bit)
        cv.imwrite(export_path+"/fusion_mertens.jpg", res_mertens_8bit)

