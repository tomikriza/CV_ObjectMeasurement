import cv2 as cv
import numpy as np
from functions import *


def Main():

    scale_percent = 30
    path = "test/1.jpg"
    
    imgOriginal = cv.imread(path)
    img = downscale(imgOriginal,scale_percent)
    
    contours = getContours(img)
    
    contoursFiltered = removeContoursSmallerThan(contours, 1000)
    paperContour = findGreatestContour(contoursFiltered)
    
    cv.drawContours(img,paperContour,-1,(0,255,0),3)

    a4Scale = 3

    width = 297 * a4Scale
    height = 210 * a4Scale
    pointsOriginal = np.float32(toPointsList(cv.approxPolyDP(paperContour,0.03 * cv.arcLength(contoursFiltered[0],True),True)))
    pointsWarped = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv.getPerspectiveTransform(pointsOriginal,pointsWarped)
    imgWarped = cv.warpPerspective(img,matrix,(width,height))

    contoursNew = getContours(imgWarped)
    contoursFiltered = removeContoursSmallerThan(contoursNew, 900)
    #cv.drawContours(imgWarped,contoursFiltered,-1,(0,255,0),3)

    cv.imshow("Warped Picture", imgWarped)
    cv.imshow("Picture", img)
    cv.waitKey(0)
    
    
Main()