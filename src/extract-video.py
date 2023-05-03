import cv2
import mediapipe as mp
import math
import time
import csv

cap = cv2.VideoCapture('./dataset_raw/Intro/Intro0.mp4')
cap.set(cv2.CAP_PROP_FPS, 100)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.8, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
start_time = time.time()

frame_counter = 0
while True:
    success, image = cap.read()
    if success:
        frame_counter += 1
        if frame_counter % 2 != 0:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    distance = math.sqrt((index_finger_tip.x - pinky_tip.x) ** 2 + (index_finger_tip.y - pinky_tip.y) ** 2)
                    current_time = time.time()
                    if current_time - start_time >= 1:
                        print(distance)
                        start_time = current_time
        cv2.waitKey(2)
        time.sleep(0.01)
    else:
        break

cap.release()
hands.close()
