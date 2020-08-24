import cv2 as cv
import numpy as np
from functions import *


def Main():

    scale_percent = 30 # percent of scaling of the original image --> for big resolution images
    path = "test/4.jpg"
    
    imgOriginal = cv.imread(path)
    img = downscale(imgOriginal,scale_percent) # scaling big picture
    
    contours = getContours(img) #find contours in the image
    
    contoursFiltered = removeContoursSmallerThan(contours, 1000) # remove insignifficant contours
    paperContour = findGreatestContour(contoursFiltered) # find contour of the A4 paper
    
    #cv.drawContours(img,paperContour,-1,(0,255,0),3)

    a4Scale = 3 # scale for adjust warped image size

    width = 297 * a4Scale
    height = 210 * a4Scale
    # points of the paper
    pointsOriginal = np.float32(toPointsList(cv.approxPolyDP(paperContour,0.03 * cv.arcLength(paperContour,True),True)))
    # new coordinates for paper
    pointsWarped = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv.getPerspectiveTransform(pointsOriginal,pointsWarped) # find transformation matrix
    imgWarped = cv.warpPerspective(img,matrix,(width,height)) # generate warped image

    # contoursNew = getContours(imgWarped)
    # contoursFiltered = removeContoursSmallerThan(contoursNew, 900)
    
    imgGrayWarped = cv.cvtColor(imgWarped,cv.COLOR_BGR2GRAY)
    imgBlurWarped = cv.GaussianBlur(imgGrayWarped,(7,7),7)
    
    imgCannyWarped = cv.Canny(imgGrayWarped,100,300)
    imgCannyWarped = cv.dilate(imgCannyWarped,(3,3))
    
    
    contours, hierarchy = cv.findContours(imgCannyWarped,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) 
    
    contours = findGreatestContour(contours)
    pointsObject = np.int32(toPointsList(cv.approxPolyDP(contours,0.03 * cv.arcLength(contours,True),True)))
    
    print(pointsObject)
    rect = cv.minAreaRect(pointsObject)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(imgWarped,[box],0,(0,0,255),2)
    #cv.imshow("Picture", img)
    cv.imshow("Warped Picture", imgWarped)
    #cv.imshow("Canny Warped", imgCannyWarped)
    cv.waitKey(0)
    
    
Main()