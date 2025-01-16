from ultralytics import YOLO
import cv2
import cvzone
import math
import torch 

# web_url = "https://192.0.0.4:8000"

cap = cv2.VideoCapture("../yolobasics/images/v5.mp4")

maskImg = cv2.imread("mask.png")
mask = cv2.resize(maskImg,(800,470))

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
    img = cv2.resize(img,(800,470))
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(img,stream=True)

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
            if currentClass =="car" or currentClass == "truck" and conf>0.9:
                cvzone.putTextRect(img,f"{classNames[cls]} {conf}",(max(0,x1),max(35,y1)),scale = 1,thickness=1,offset=3)
                cvzone.cornerRect(img,(x1,y1,w,h),l=9)

    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(1)