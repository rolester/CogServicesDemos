#pip install opencv-python==3.4.3.18
from __future__ import print_function
import cv2 as cv2
import argparse
import json
import numpy as np

devicename = 0

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0] #hew
    s = hsv[:,:,1] #saturation
    v = hsv[:,:,2] #intensity

    ret, min_sat = cv2.threshold(s,35,255, cv2.THRESH_BINARY) # Anything value 35 or higher will be white
    ret, max_hue = cv2.threshold(h,20, 255, cv2.THRESH_BINARY_INV)
    final = cv2.bitwise_and(min_sat, max_hue)
    frame = cv2.bitwise_and(frame, frame, mask=final)

    cv2.imshow('Capture - Face detection', frame)

camera_device = devicename
#-- 2. Read the video stream
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    if cv2.waitKey(10) == 27:
        break
