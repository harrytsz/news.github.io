## 手势识别 


### Gesture Recongnition

手势识别功能主要是通过 Google MediaPipe 库进行二次开发，通过 OpenCV 库驱动终端摄像头进行实时捕捉手部图像，根据手部关节点之间的空间关系自定义各种不同的触发动作。

![](../style/gestures.gif)

![](https://pic.imgdb.cn/item/61442ab12ab3f51d91972de8.jpg)

手势识别功能主要分为两部分，一部分将 Mediapipie 接口进行更高层次的封装。代码如下：

```python
"""
    Hand Tracking Module
    By: Harrytsz
"""
import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
```

手势识别的第二部分，通过各个手指关节点之间的空间关系定义触发动作。比如，当只伸出食指时，可以定义鼠标跟踪食指移动。

![](https://pic.imgdb.cn/item/61ce8d622ab3f51d9175d277.png)

当检测出食指和拇指之间的距离小于阈值时，触发鼠标点击事件。还有很多功能可以自定义。

![](https://pic.imgdb.cn/item/61ce8d3a2ab3f51d9175ab9a.png)

代码如下：

```python
import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy

#######################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 8  # 调节鼠标移动灵敏度
#######################
pTime = 0
# 平滑处理，防止鼠标抖动不方便点击操作
plocX, plocY = 0,0
clocX, clocY = 0,0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # 获取屏幕尺寸  1920 * 1080

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
        # 4. Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Covert Coordicates
            # 鼠标全屏移动
            # x3 = np.interp(x1, (0, wCam), (0, wScr))
            # y3 = np.interp(y1, (0, hCam), (0, hScr))
            # 鼠标在框中移动
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # 6. Smooothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # 7. Move Mouse
            # autopy.mouse.move(x3,y3)
            # autopy.mouse.move(wScr-x3,y3)
            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8. Both Index and middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8,12,img)
            # 10. Click mouse if distance short
            if length < 25:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 13. DIY 自由发挥
        if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # 如果大拇指和食指的距离小于阈值，触发按住鼠标左键
            length, img, lineInfo = detector.findDistance(4, 8, img)
            if length < 30:
                autopy.mouse.click()
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
```

至此，手势控制功能已经完成。新闻推荐系统的用户可以通过手势控制新闻页面的切换操作，用户还可以通过调节参数来控制手势的灵敏度。例如通过调节 smoothening 参数就可以改变鼠标的移动平滑程度。