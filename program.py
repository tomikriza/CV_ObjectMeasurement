import cv2 as cv
import numpy as np
from functions import *


def Main():
    width, height = 640,480
    scale_percent = 30
    path = "test/1.jpg"
    
    imgOriginal = cv.imread(path)
    img = downscale(imgOriginal,scale_percent)
    
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgGray = cv.GaussianBlur(imgGray,(7,7),5)
    imgCanny = cv.Canny(img,50,100)
    
    contours, hierarchy = cv.findContours(imgCanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    contoursFiltered = removeContoursSmallerThan(contours, 400)
    
    list_of_points = toTupleList(cv.approxPolyDP(contoursFiltered[0],0.03 * cv.arcLength(contoursFiltered[0],True),True))

    #cv.drawContours(img, contoursFiltered, -1, (0,255,0), 3)
    
    
    print(list_of_points)  
    
    #cv.imshow("Picture", img)
    cv.waitKey(0)
    
    
Main()