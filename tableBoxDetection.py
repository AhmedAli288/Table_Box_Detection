import cv2
import numpy as np

# This function takes a cv2.imread returned file and returns coordinates of boxes

def detect_box(image, line_min_width=55):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh,img_bin=cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernal_h=np.ones((1,line_min_width), np.uint8)
    kernal_v=np.ones((line_min_width,1), np.uint8)
    img_bin_h=cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v=cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal_v)
    img_bin_final=img_bin_h|img_bin_v
    final_kernel=np.ones((3,3), np.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=2)
    ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=4, ltype=cv2.CV_32S)
    return stats,labels