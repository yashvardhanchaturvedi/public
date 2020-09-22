from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
from imutils import face_utils
import imutils
import time
import cv2
import pyzbar.pyzbar as pyzbar
import math
import urllib.request
import os
import time
import dlib

vs=1
net=1
fps=1
CLASSES=[]
COLORS=[]
deti="no"

flag=1

file="sound.mp3"
path='shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(path)

t1=0
flag=1

class VideoCamera(object):
    def __init__(self):
        global vs,deti
        global CLASSES,fps,net,COLORS
        
        vs = VideoStream(src=0).start()
        

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]

        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

            # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")

            # initialize the video stream, allow the cammera sensor to warmup,
            # and initialize the FPS counter
        print("[INFO] starting video stream...")
            #vs = VideoStream(src=0).start()
        #time.sleep(2.0)
        fps = FPS().start()

#Function for carrying out all the processes on a single frame
    def get_frame(self,shapeof,bound_lowX=100,bound_highX=300,bound_lowY=50,bound_highY=250,ptX=200,ptY=150,radius=100,detect="no",license="no",start="no",helmet_detect="no"):
        global vs,flag
        global t1,flag

        stri="No message"
        qr="No qr code detected"

        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

        #To check if the bottom 2 points of the detected square lie in the desired circle(authorized area)
        def check_value(startX,startY,endX,endY,ptX,ptY,radius):
            val=(startX-ptX)**2 + (endY-ptY)**2
            if(val-radius**2>0):
                return -1
            val=(endX-200)**2 + (endY-ptY)**2
            if(val-radius**2>0):
                return -1
            return 1
        frame = vs.read()
        
        frame = imutils.resize(frame, width=400)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        net.setInput(blob)
        detections = net.forward()
        boundaries_y = [([0, 100, 160], [70, 255, 255])]
        boundaries_r = [([0, 60, 180], [100, 120, 255])]
        font = cv2.FONT_HERSHEY_SIMPLEX

        def detectcolor(boundaries, img):
            for (lower, upper) in boundaries:
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                mask = cv2.inRange(img, lower, upper)
                output = cv2.bitwise_and(img, img, mask = mask)
            ncol=0
            for i in range(0, output.shape[0]):
                for j in range(0, output.shape[1]):
                    if(output[i][j][0]>0 or output[i][j][1]>0 or output[i][j][2]>0):
                        ncol=ncol+1
            colorperc=round(ncol/(img.shape[1]*img.shape[0])*100,2)
            return([colorperc,output])

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.20:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                #For license plate detection
                if(license=="yes"):
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                #Qr code detection
                decodedObjected=pyzbar.decode(frame)
                for obj in decodedObjected:
                    qr=obj.data.decode('utf-8')
                    (a,b,c,d)=obj.rect
                    cv2.rectangle(frame,(a,b),(a+c,b+d),(255,255,0),3)
                #To display the rectangles only if human is detected
                if(idx==15 and start=="yes"):
                    #If boundary is a square  
                    if(shapeof=="square"):
                        cv2.rectangle(frame,(bound_lowX,bound_lowY),(bound_highX,bound_highY),(255,0,0),2)
                        if(startX<bound_lowX or endY<bound_lowY or endY>bound_highY or endX>bound_highX): 
                            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)
                            if((endX-startX)*(endY-startY)):
                                y = startY - 15 if startY - 15 > 15 else startY + 15
                                cv2.putText(frame, "Authorized "+label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                                stri="Authorized area, person found"
                        else:
                            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255), 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, "Unauthorized " +label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            stri="UnAuthorized area,person found! PLease checK!"
                    #If boundary is a circle
                    else:
                        cv2.circle(frame,(ptX,ptY),radius,(255,0,0),2)
                        value=check_value(startX,startY,endX,endY,ptX,ptY,radius)
                        if(value==-1):
                            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, "Authorized "+label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            stri="Unauthorized area,person found! PLease checK!"
                        else:
                            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255), 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, "Authorized " +label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            stri="Authorized area, person found"
                #For helment detection
                if(helmet_detect=="yes"): 
                    h=int((endX-startX)*0.3)
                    w=(endX-startX)
                    if(endX-startX>20):
                        Forehead=frame[startY:startY+h, startX:startX+w]
                        det_y=detectcolor(boundaries_y, Forehead)
                        det_r=detectcolor(boundaries_r, Forehead)
                        if(det_y[0]>1 or det_r[0]>1):
                            cls="Helmet"
                            R=0
                            G=255
                        else:
                            cls="Head"
                            R=255
                            G=0
                        frame = cv2.rectangle(frame, (startX, startY), (startX+w,startY+h), (0,G,R), 5)
                        cv2.putText(frame, cls+":"+str(round(confidence*100,2))+"%", (startX,startY-5), font, 1, (0, G, R), 4, cv2.LINE_AA)
                #For face detection
                if(detect=="yes"):
                        global deti
                        rects = detector(gray, 0)

                        for rect in rects:

                                # determine the facial landmarks for the face region, then

                                # convert the facial landmark (x, y)-coordinates to a NumPy

                                # array

                                shape = predictor(gray, rect)

                                shape = face_utils.shape_to_np(shape)

                                y=shape[45][1]-shape[36][1]

                                x=shape[45][0]-shape[36][0]

                                if(0.01*x>=y):
                                                #print("IN loop")

                                                R=0

                                                G=255

                                                B=0
                                                #To capture image in case a face is found
                                                frame1=frame[max(shape[20][1]-int(x*1.2),0):min(shape[9][1]+int(x*0.1),frame.shape[0]),max(shape[1][0]-int(x*0.2),0):min(shape[16][0]+int(x*0.2),frame.shape[1])]
                                                cv2.imwrite("face_64op/" + str(t1)+".jpg",frame1)
                                                deti="no"
                                                if(flag==1):
                                                    os.system("mpg123 " + file) #Sound played when face is captured
                                                    flag=0

                                                

                                else:

                                                R=255

                                                G=0

                                                B=0

                                # loop over the (x, y)-coordinates for the facial landmarks



                                for (x, y) in shape:

                                                cv2.circle(frame, (x, y), 1, (B, G, R), -1)            #cv2.rectangle(frame,(startX,startY),(int(endX/5),int(endY/5)),(255,0,0),2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(),stri,qr,deti #Returns the frame,text,qrcode