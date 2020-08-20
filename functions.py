import cv2 as cv
import numpy as np

# Returns downscaled image scale_precent % img 
def downscale(img, scale_percent):
    while (scale_percent >= 100 ):
        print("Input scale factor value lesser than 100")
        scale_percent = int(input())
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width,height)
    resized = cv.resize(img, dim,interpolation=cv.INTER_AREA)
    return resized

# Returns list of contours with area greater than threshold_area
def removeContoursSmallerThan(contours, threshold_area):
    contoursFiltered = []
    for cnt in contours:
        if cv.contourArea(cnt) > threshold_area:
            contoursFiltered.append(cnt)
    return contoursFiltered

# Returns list of points from approxPolyDP
def toTupleList(list):
    points = []
    for point in list:
        points.append((point[0][0],point[0][1]))
    return points