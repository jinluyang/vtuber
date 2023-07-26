import os
import cv2
import time


IMG_SIZE = (512, 512)
img_folder = './resources'
while(True):
    for img_file in os.listdir(img_folder):
        img_path = img_folder + '/' + img_file
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=IMG_SIZE)
        cv2.imshow('cur_img', img)
        cv2.waitKey(5000)