import cv2
import mediapipe as mp 
import time 

cap=cv2.VideoCapture("Video/3.mp4")

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

pTime = 0
while True:
    sucess,img = cap.read()
    img = cv2.resize(img,(820,600))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id , detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)
            # print(id,detection)
            # print(detection.location_data.RelativeBoundingBox)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=(
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'{int(detection.score[0]*100)} %',
                        (bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            

                 


    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS: {str(int(fps))}',(50,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(20)