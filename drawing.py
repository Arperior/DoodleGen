import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import HandTracking as htm

folderPath = "Header"
myList = os.listdir(folderPath)
overlay = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlay.append(image)

header = overlay[0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,728)

detector = htm.handDetector(detectioncon=0.8)

#imgCanvas = np.full((720,1280,3),255,dtype=np.uint8)
imgCanvas = np.zeros((720,1280,3),np.uint8)
drawcolour = (0,0,255)
brushThickness = 15
xp,yp = 0,0

while(True):
    success,img = cap.read()
    img = cv2.flip(img,1)

    img = detector.findhands(img)
    lmList = detector.findpos(img,draw=False)

    if len(lmList) != 0:
            #Index Finger Tip  
            x1,y1 = lmList[8][1:]
            #Midle Finger Tip
            x2,y2 = lmList[12][1:]

    fingers = detector.fingersUp()
    fist = detector.fistC()

    if len(fingers) != 0:
        if fingers[1] and fingers[2]:
             xp,yp = 0,0
             if y1 < 125:
                  if 250<x1<450:
                       header = overlay[0]
                       drawcolour = (0,0,255)
                  elif 550<x1<750:
                       header = overlay[1]
                       drawcolour = (255,0,0)
                  elif 800<x1<950:
                       header = overlay[2]
                       drawcolour = (0,255,0)
                  elif 1050<x1<1200:
                       header = overlay[3]  
                       drawcolour = (0,0,0)
             #print("selection Mode")
             cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawcolour,cv2.FILLED)
                  
        if fingers[1] and fingers[2] == False:
             cv2.circle(img,(x1,y1),15,drawcolour,cv2.FILLED)
             #print('Drawing Mode')
             if xp ==0 and yp == 0:
                  xp,yp = x1,y1
             cv2.line(img,(xp,yp),(x1,y1),drawcolour,thickness=brushThickness)
             cv2.line(imgCanvas,(xp,yp),(x1,y1),drawcolour,thickness=brushThickness)

             xp,yp = x1,y1
        
        if fist:
             cv2.circle(img,(x1,y1),40,drawcolour,cv2.FILLED)
             cv2.circle(imgCanvas,(x1,y1),40,drawcolour,cv2.FILLED)

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    img[0:125,0:1280] = header
    #cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow('Image',img)
    cv2.imshow("Virtual canvas",imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

