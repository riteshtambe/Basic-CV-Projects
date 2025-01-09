import cv2
import numpy as np
import time 
import os
import HandTrckingModules as htm 

brushThickness = 15
eraserThickness = 50

folderPath = "C:\MyWorkSpace\Courses\Computer Vision\Hand Tracking Project\paintphotos"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    # resized_overlay = cv2.resize(image,(1280,125))
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255,0,0)
xp,yp = 0,0
# imgCanvas = np.zeros((720,1280,3),np.uint8)
imgCanvas = np.zeros((700, 1280, 3), np.uint8)
# url = "http://192.0.0.4:8080/video"
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1. Import Image
    success , img = cap.read()
    img = cv2.resize(img,(1280,700))
    img = cv2.flip(img,1)

    # 2. Find Hand LandMarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        # print(lmList)

        #Tip of index and middle finger 
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

    # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),(255,0,255),cv2.FILLED)
            print("Selection Mode")

            #Checking For the Click 
            if y1<125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                if 550<x1<750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                if 800<x1<950:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                if 1050<x1<1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

    # 5. If Drawing Mode - Index finger is Up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            print("Drawing Mode")

            if xp==0 and yp==0:
                xp,yp = x1,y1
            
            #Increses the thickness of Eraser
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    imgCanvas = cv2.resize(imgCanvas, (1280, 700))  # Resize imgCanvas
    imgInv = cv2.resize(imgInv, (1280, 700))  # Resize imgInv
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    img[0:125,0:1280] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)
    

