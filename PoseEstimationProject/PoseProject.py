import cv2
import mediapipe as mp
import time
import PoseEstimatonModule as pm


try:
    cap = cv2.VideoCapture("videos/5.mp4")
    cTime = 0
    pTime = 0
    detector = pm.poseDetector()
    while True:
        success,img = cap.read()
        img = cv2.resize(img,(720,720))

        img=detector.findPose(img,draw=True)
        lmList = detector.findPositon(img,draw=False)
        cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(255,0,0) , cv2.FILLED)
        print(lmList)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
        
        cv2.imshow("Image",img)
        cv2.waitKey(1)

except Exception as e:
    print(e)
    
    


