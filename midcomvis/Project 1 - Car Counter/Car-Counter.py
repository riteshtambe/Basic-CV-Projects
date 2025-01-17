from ultralytics import YOLO
import cv2
import cvzone
import math
import torch 
from sort import * # Tracker Import 
# web_url = "https://192.0.0.4:8000"

cap = cv2.VideoCapture("../yolobasics/images/v4.mp4")

maskImg = cv2.imread("mask4.png")
mask = cv2.resize(maskImg,(800,470))

#Tracking Code  - Object Counter Companion code
tracker = Sort(max_age = 20, min_hits=2,iou_threshold=0.3) 
limits = [310,297,673,297] #cv2.line limit 
totalCounts = []

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
              "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", 
              "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
              "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", 
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("../midcomvis/YOLO-Weights/yolov8n.pt")
model.to(device)
# model.train(data="coco8.yaml",epochs=100,imgsz=640,device=0)

while True:
    success , img = cap.read()
    if not success:
        break
    img = cv2.resize(img,(800,470))
    imgRegion = cv2.bitwise_and(img,mask)
    
    

    imgGraphics = cv2.imread("graphics.jpg",cv2.IMREAD_UNCHANGED)

    if imgGraphics.shape[2]==4:
        img = cvzone.overlayPNG(img,imgGraphics,(0,0))
    results = model(imgRegion,stream=True)

    detections = np.empty((0,5)) #Tracking Code  - Object Counter Companion code

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding Boxes
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w,h = x2-x1,y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h),l=9)

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100

            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass =="car" and conf>0.6:
                cvzone.putTextRect(img,f"{classNames[cls]} {conf}",(max(0,x1),max(35,y1)),
                scale = 1,thickness=1,offset=3)
                # cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)

                #Tracking Code  - Object Counter Companion code
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    resultsTracker=tracker.update(detections)

    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,Id = map(int,result)
        # print(Id)
        print(result) # Objet Counter
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5,colorR=(255,0,0))
        cvzone.putTextRect(img,f"ID {Id}",(max(0,x1),max(35,y1)),
                scale = 1,thickness=1,offset=3)

        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-20 < cy < limits[1]+20:
            if totalCounts.count(Id)==0:
                 totalCounts.append(Id)
                 cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)

                 


    cvzone.putTextRect(img,f"Count:{len(totalCounts)}",(50,50))

    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(1)