import cv2 as cv
import numpy as np
from functions import *

def empty():
    pass


frameWidth = 640
frameGeight = 480
path = "test/1.jpg" ### Enter custom path
img = downscale(cv.imread(path),30) 

windowName = "TrackBars"

cv.namedWindow(windowName)
cv.resizeWindow(windowName,500,200)
cv.createTrackbar("Threshold lower",windowName,0,1500,empty)
cv.createTrackbar("Threshold upper",windowName,500,1500,empty)

while True:
    threshold_lower = cv.getTrackbarPos("Threshold lower",windowName)
    threshold_upper = cv.getTrackbarPos("Threshold upper",windowName)

    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(7,7),5)
    imgCanny = cv.Canny(imgGray,threshold_lower,threshold_upper)

    cv.imshow("Canny Image",imgCanny)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
