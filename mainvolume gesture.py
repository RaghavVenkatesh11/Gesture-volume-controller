import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import pyttsx3

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Volume setup
devices = AudioUtilities.GetSpeakers()

interface = devices._dev.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volumeCtrl = cast(interface, POINTER(IAudioEndpointVolume))

minVol, maxVol = volumeCtrl.GetVolumeRange()[0:2]

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

lastVolume = -1
lastBrightness = -1

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:

        for i, handLms in enumerate(results.multi_hand_landmarks):

            label = results.multi_handedness[i].classification[0].label

            lmList = []

            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):

                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append((cx, cy))

            if len(lmList) >= 9:

                # Thumb tip
                x1, y1 = lmList[4]

                # Index tip
                x2, y2 = lmList[8]

                # Distance
                length = np.hypot(x2 - x1, y2 - y1)

                # Draw fingers
                cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # RIGHT HAND = VOLUME
                if label == "Right":

                    vol = np.interp(length, [30, 200], [minVol, maxVol])

                    volPerc = int(
                        np.interp(length, [30, 200], [0, 100])
                    )

                    volumeCtrl.SetMasterVolumeLevel(vol, None)

                    cv2.putText(
                        img,
                        f"Volume: {volPerc}%",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3
                    )

                    cv2.rectangle(
                        img,
                        (50, 70),
                        (50 + volPerc * 2, 100),
                        (0, 255, 0),
                        cv2.FILLED
                    )

                    if abs(volPerc - lastVolume) >= 10:
                        engine.say(f"Volume {volPerc} percent")
                        engine.runAndWait()
                        lastVolume = volPerc

                # LEFT HAND = BRIGHTNESS
                elif label == "Left":

                    brightness = int(
                        np.interp(length, [30, 200], [0, 100])
                    )

                    sbc.set_brightness(brightness)

                    cv2.putText(
                        img,
                        f"Brightness: {brightness}%",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        3
                    )

                    cv2.rectangle(
                        img,
                        (50, 170),
                        (50 + brightness * 2, 200),
                        (255, 255, 0),
                        cv2.FILLED
                    )

                    if abs(brightness - lastBrightness) >= 10:
                        engine.say(f"Brightness {brightness} percent")
                        engine.runAndWait()
                        lastBrightness = brightness

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    cv2.imshow("WaveTune - Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
