import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
from ultralytics import YOLO
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("C:/Users/darri/OneDrive/Desktop/Work/Projects/Vehicle Detection/Attachments/video.mp4")

frame_width = 640
frame_height = 480

model = YOLO("C:/Users/darri/OneDrive/Desktop/Work/Projects/Vehicle Detection/yolov5mu.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "cat", "dog",
              "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("C:/Users/darri/OneDrive/Desktop/Work/Projects/Vehicle Detection/Attachments/mask.png")
mask = cv2.resize(mask, (frame_width, frame_height))

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []

lineCord = [200, 297, 550, 297]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (frame_width, frame_height))
    maskReg = cv2.bitwise_and(img, mask)

    results = model(maskReg, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width - 1, x2)
            y2 = min(frame_height - 1, y2)

            w = x2 - x1
            h = y2 - y1

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            conf = math.ceil((box.conf[0].item() * 100)) / 100
            if currentClass in ["car", "motorbike", "truck", "bus"] and conf > 0.35:
                cvzone.cornerRect(img, (x1, y1, w, h),t=1,rt=1,l=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 255),t=3,rt=3,l=10)

        cx = x1 + w // 2
        cy = y1 + h // 2
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), cv2.FILLED)

        if lineCord[0] < cx < lineCord[2] and lineCord[1] - 5 < cy < lineCord[3] + 5:
            if Id not in totalCount:
                totalCount.append(Id)

    cv2.line(img, (lineCord[0], lineCord[1]), (lineCord[2], lineCord[3]), (240, 255, 20), 3)
    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (18, 35), scale=1.5, thickness=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()