import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture('./src/exampleVideos/test_video.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils
p_time = 0

while True:
    success, image = cap.read()
    if success:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                distance = math.sqrt((index_finger_tip.x - pinky_tip.x) ** 2 + (index_finger_tip.y - pinky_tip.y) ** 2)
                print(distance)

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("cv2", image)
        cv2.waitKey(1)
    else:
        break

cap.release()
