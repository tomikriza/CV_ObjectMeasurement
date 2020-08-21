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

# Returns list of points from return value of approxPolyDP
def toPointsList(list):
    points = []
    for point in list:
        points.append([point[0][0],point[0][1]])
    points[1], points[2], points[3] = points[3], points[1], points[2]
    return points

def getContours(img):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgGray = cv.GaussianBlur(imgGray,(9,9),5)
    imgCanny = cv.Canny(img,50,100)
    contours, hierarchy = cv.findContours(imgCanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    return contours

def findGreatestContour(contours):
    greatestCnt = contours[0]
    for cnt in contours:
        if cv.contourArea(cnt) >= cv.contourArea(greatestCnt):
            greatestCnt = cnt
    return greatestCnt