import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False,maxHands=2,modelComp=1,detectioncon=0.5,trackcon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        self.modelComp = modelComp

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectioncon,
            min_tracking_confidence=self.trackcon,
            model_complexity = self.modelComp)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findpos(self,img,handno=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                #print(id,":",cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    if id == 0:
                        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)

    ptime = 0
    ctime = 0

    detector = handDetector()

    while(True):
        success,img = cap.read()

        img = detector.findhands(img)
        lmlist = detector.findpos(img)

        if len(lmlist) != 0:
            print(lmlist[4])

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow('Image',img)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break


if __name__ == '__main__':
    main()
