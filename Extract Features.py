from ultralytics import YOLO
import cv2
import pandas as pd

# load the pose estmation YOLO model

model=YOLO("yolov8n-pose.pt")

# conert the path and of video name  want to extract features from it

cap=cv2.VideoCapture('Violance.mp4')

data=[]

# loop to capture  the frame
while cap.isOpened():

    # read frame
    ret,frame=cap.read()
    
    # if not frame exit
    if not ret:

        break

    #resize the frame
    frame=cv2.resize(frame,(600,500))

    # found the boxes
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    # found the keypoint
    keypoint=result[0].keypoints.data.cpu().numpy().astype("float")

    # id and keypoint for each body
    for idd,lm in enumerate(keypoint):
        if lm.shape[0]>0:
            k = []
            #convert lm from 2D to 1D
            for l in range(len(lm)) :
                k.append(lm[l][0])
                k.append(lm[l][1])
                k.append(lm[l][2])

            data.append(k)
            print(k)



    # Show frame
    cv2.imshow("frame",frame)
    cv2.waitKey(1)
# put  total data into DataFrame
new=pd.DataFrame(data)
# Save it in csv file (put your path want to save)
new.to_csv("walk.csv")
cap.release()
cv2.destroyAllWindows()