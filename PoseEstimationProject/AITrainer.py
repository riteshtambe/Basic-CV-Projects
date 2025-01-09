import cv2
import numpy as np
import mediapipe as mp
import time
import PoseEstimatonModule as pm


cap = cv2.VideoCapture("trainervideos/1.mp4")

detector = pm.poseDetector()


while True:
    # success , img = cap.read()
    # img = cv2.resize(img,(720,720))
    img = cv2.imread("trainervideos/7.jpg")
    img = detector.findPose(img)
    lmList = detector.findPositon(img,False)

    if len(lmList)!=0:
        detector.findAngle(img)

    cv2.imshow("Image",img)
    cv2.waitKey(1)