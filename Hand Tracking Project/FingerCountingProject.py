
import cv2
import os
import time
import HandTrckingModules as htm
# Set up video capture
cap = cv2.VideoCapture(0)

folderPath = "C:\MyWorkSpace\Courses\Computer Vision\Hand Tracking Project\Hand Photos"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    resized_overlay = cv2.resize(image, (200, 200)) 
    overlayList.append(resized_overlay)

detector = htm.handDetector(detectionCon=0.75)

# Target frame size (fixed size of the video window)

pTime=0
tipIds = [4,8,12,16,20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    
    if len(lmList)!=0:
        fingers = []

        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #fOUR fINGERS
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        
        totalFingers = fingers.count(1)
        print(totalFingers)


        h,w,c = overlayList[totalFingers-1].shape
        img[0:h,0:w]=overlayList[totalFingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{str(totalFingers)}",(65,375),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),15)

    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    # cv2.rectangle(img,(,400),(400,70),(255,0,0),cv2.FILLED)
    cv2.putText(img,f"FPS: {int(fps)}",(400,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



