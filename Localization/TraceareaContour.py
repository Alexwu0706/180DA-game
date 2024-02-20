import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Setting Webcab size
cap = cv.VideoCapture(0)
def areaofCnt(img_mask,imgContour,color,x,y):
    contours,hierarchy = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(imgContour,contours, -1, (255,0,255),7)
    for cnt in contours:
        area = cv.contourArea(cnt) 
        if((area > 5000) and (area <= 10000)):
            cv.putText(imgContour, "area is: " + str(area), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(imgContour, "color is: " + color, (x+50,y+50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv.putText(imgContour, "close", (300,300), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif(area > 10000):
            cv.putText(imgContour, "area is: " + str(area), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(imgContour, "color is: " + color, (x+50,y+50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv.putText(imgContour, "open", (300,300), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

while(1):
    # Take each frame (1)
    _, frame = cap.read()  
    imgContour = frame.copy()
    # define range of blue color in HSV
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    lower_green = np.array([50,100,100])
    upper_green = np.array([70,255,255])
    lower_blue = np.array([95,100,100])
    upper_blue = np.array([115,255,255])

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors (2)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image (3)
    res = cv.bitwise_and(frame, frame, mask= mask)
    
    #frame[y1:y2, x1:x2]
    #frame width = 640      frame height = 480
    left_screen = frame[300:480,250:400]
    #left_screen = frame
    crop_hsv = cv.cvtColor(left_screen, cv.COLOR_BGR2HSV)
    #mask_interest1 = cv.inRange(crop_hsv,lower_yellow, upper_yellow)
    mask_interest1 = cv.inRange(crop_hsv,lower_yellow, upper_yellow)
    mask_interest2 = cv.inRange(hsv,lower_blue, upper_blue)

    areaofCnt(mask_interest1,imgContour,"yellow",100,100)
    areaofCnt(mask_interest2,imgContour,"blue",300,300)

    # Display of your usual four images
    cv.imshow('imgContour',imgContour)
    #cv.imshow('mask',mask)
    #cv.imshow('res',res)
    #cv.imshow('crop',img)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()