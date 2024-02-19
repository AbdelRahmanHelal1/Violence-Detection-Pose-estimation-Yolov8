from ultralytics import YOLO
import cv2
import numpy as np
import pickle
###############
model1=pickle.load(open("voilance-model.h5","rb"))

##########
model=YOLO("yolov8n-pose.pt")

cap=cv2.VideoCapture(r'E:\windows\cpv\New folder (3)\walk.mp4')
data=[]
while cap.isOpened():

    ret,frame=cap.read()
    if not ret:

        break
    frame=cv2.resize(frame,(600,500))
    result=model.predict(frame)
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    keypoint=result[0].keypoints.data.cpu().numpy().astype("float")

    for idd,lm in enumerate(keypoint):

        if lm.shape[0]>0:
           for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                k = []
                for l in range(len(lm)) :

                    k.append(lm[l][0])
                    k.append(lm[l][1])
                    k.append(lm[l][2])
                reshaped_list = np.array([k])
                if model1.predict(reshaped_list)[0]==1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame,"Fighting",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

    cv2.imshow("frame",frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()