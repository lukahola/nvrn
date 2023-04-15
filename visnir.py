import numpy as np
import cv2 as cv



def name_config(img_file, img_id):
    rgb_img = img_id + '_rgb.tiff'
    nir_img = img_id + '_nir.tiff'
    rgb_file = img_file + rgb_img
    nir_file = img_file + nir_img
    return rgb_file, rgb_img, nir_file, nir_img

def show_img(img_path, img_name):
    Img = cv.imread(img_path)
    cv.imshow(img_name, Img)
    cv.waitKey(0)
    return Img

# def sift_img(img):
#

if __name__ == '__main__':

    file = 'G:/Datasets/nirscene1/indoor/'
    id = '0006'
    rgb_file, rgb_img, nir_file, nir_img = name_config(file, id)
    RGB = cv.imread(rgb_file)
    NIR = cv.imread(nir_file)
    # imgs = np.hstack([RGB, NIR])
    # cv.imshow('imgs', imgs)
    # cv.waitKey(0)

    sift = cv.xfeatures2d.SIFT_create
    kp1, des1 = sift.detectAndCompute(RGB, None)
    kp2, des2 = sift.detectAndCompute(NIR, None)
    cv.drawKeypoints(RGB, kp1, RGB, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.drawKeypoints(NIR, kp2, NIR, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgs = np.hstack([RGB, NIR])
    cv.imshow('sift', imgs)
    cv.waitKey(0)