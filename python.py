import cv2
import mediapipe as mp
import math
from pyautogui import hotkey, press
import time


class HandDetector:
    def _init_(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        if draw:
            return allHands, img
        else:
            return allHands, img

    def fingersUp(self, myHand):
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    swipe_ended = False
    swipe_started = False
    lswipe_ended = False
    lswipe_started = False
    zoom_started = False
    zoom_ended = False
    zout_started = False
    zout_ended = False
    left_threshold = 400
    right_threshold = 270
    start_time = 0

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1["center"]
            handType1 = hand1["type"]

            fingers1 = detector.fingersUp(hand1)

            if (
                fingers1[0]
                and fingers1[1]
                and not fingers1[2]
                and not fingers1[4]
                and not fingers1[3]
            ):
                length, info, _ = detector.findDistance(
                    lmList1[4][0:2], lmList1[8][0:2], img
                )
                if not zoom_started and length < 50:
                    zoom_started = True

                if zoom_started and length > 120:
                    zoom_started = False
                    zoom_ended = True

                if zoom_ended:
                    zoom_ended = False
                    print("Zoomed")
                    hotkey("ctrl", "+")

            if (
                fingers1[0]
                and fingers1[1]
                and fingers1[2]
                and not fingers1[3]
                and not fingers1[4]
            ):
                length, info, _ = detector.findDistance(
                    lmList1[4][0:2], lmList1[8][0:2], img
                )
                if not zout_started and length > 120:
                    zout_started = True

                if zout_started and length < 50:
                    zout_started = False
                    zout_ended = True

                if zout_ended:
                    zout_ended = False
                    print("Zoomed Out")
                    hotkey("ctrl", "-")

            if fingers1[1] and fingers1[2] and fingers1[3] and fingers1[4]:
                indMid = [lmList1[8][0], lmList1[12][0], lmList1[16][0], lmList1[16][0]]

                if not swipe_started and max(indMid) > left_threshold:
                    swipe_started = True

                if swipe_started and min(indMid) < right_threshold:
                    swipe_started = False
                    swipe_ended = True

                if swipe_ended:
                    print("Swipe detected!")
                    press("right")
                    swipe_ended = False

            if fingers1[1] and fingers1[2] and fingers1[3] and not fingers1[4]:
                indMid = [lmList1[8][0], lmList1[12][0], lmList1[16][0], lmList1[16][0]]

                if not lswipe_started and min(indMid) < right_threshold:
                    lswipe_started = True

                if lswipe_started and max(indMid) > left_threshold:
                    lswipe_started = False
                    lswipe_ended = True

                if lswipe_ended:
                    print("Left Swipe detected!")
                    press("left")
                    lswipe_ended = False

            if all(fingers1):
                if start_time == 0:
                    start_time = time.time()
                current_time = time.time()

                elapsed_time = current_time - start_time

                if elapsed_time >= 8:
                    print("Exiting...")
                    hotkey("alt", "left")

                    exit()

        else:
            start_time = 0

        # cv2.imshow("frame", img)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "_main_":
    main()