### This code get input images and genrate bounding box
import glob
import os
import cv2
import numpy as np
from glob import glob

input_images_path = "/media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/dataset_anson/share/XLF-0.2/input1"


list_files = glob(os.path.join(input_images_path, "*"))

for file in list_files:
    img_org = cv2.imread(file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img_org, dtype=np.uint8)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    height, width, channel = img.shape
    debug_im = np.ones((height, width, 3), np.uint8) * 255
    debug_im[:, :, 0] = img[:, :, 0]
    debug_im[:, :, 1] = img[:, :, 1]
    debug_im[:, :, 2] = img[:, :, 2]
    overlay = debug_im.copy()
    _, thresh = cv2.threshold(overlay, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
