
import cv2 as cv
import numpy as np

def sampleWithMyPic():
    x = "Images/Wikipedia/img0.jpg"
    img_fn = ["Images/t1/IMG_5035.JPG",
              "Images/t1/IMG_5036.JPG",
              "Images/t1/IMG_5037.JPG"]
    img_list = [cv.imread(fn) for fn in img_fn]
    exposure_times = np.array([0.025, 0.008, 0.0333], dtype=np.float32)

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
    cv.imwrite("ldr_debevec_X.jpg", res_debevec_8bit)
    cv.imwrite("ldr_robertson_X.jpg", res_robertson_8bit)
    cv.imwrite("fusion_mertens_X.jpg", res_mertens_8bit)



def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def change_array_to_pic(array,width):
    height = array.shape[0]
    img = array.reshape((height,1))
    img = np.tile(img, (1, width))
    img = img.transpose()
    img = img.reshape((width, height, 1))
    img = np.tile(img, (1, 1, 3))
    return img

if __name__ == '__main__':


    '''
    prtPx = np.zeros((3000,2000))
    for i in range (3000):
        for j in range(2000):
            prtPx[i,j] = int(round(255*(i/3000)*(j/2000)))

    '''
    linearImg = np.zeros((3000))
    for i in range (3000):
        linearImg[i]= int(round(255 * (i / 3000)))
    gamma_c = linearImg.copy()
    linearImg = change_array_to_pic(linearImg,2000)
    cv.imwrite("linear.jpg", linearImg)

    gamma_c = gamma_c/255
    gamma_c = gamma_c**(1.2)
    gamma_c = gamma_c*255
    gamma_c = change_array_to_pic(gamma_c, 2000)
    cv.imwrite("gamma.jpg", gamma_c)
    print("end")