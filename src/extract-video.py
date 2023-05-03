import cv2
import mediapipe as mp
import math
import time
import csv
import os

cap = cv2.VideoCapture('./dataset_raw/Intro/Video/Intro0.mp4')
cap.set(cv2.CAP_PROP_FPS, 100)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.8, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
start_time = time.time()

videos_dir = './dataset_raw/Intro/Video/'
output_file = 'finger_chords.csv'


data = [('filename','time', 'index', 'middle', 'ring', 'pinky')]

for filename in os.listdir(videos_dir):
   
    video_path = os.path.join(videos_dir, filename)
    print(f'Processing video: {video_path}')

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 100)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    start_time = time.time()

    # Loop over all frames in the video
    counter = 0
    frame_counter = 0
    while True:
        success, image = cap.read()
        if success:
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == 'Right':
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                        index_distance = math.sqrt((wrist.x - index_finger_tip.x) ** 2 + (wrist.y - index_finger_tip.y) ** 2)
                        middle_distance = math.sqrt((wrist.x - middle_finger_tip.x) ** 2 + (wrist.y - middle_finger_tip.y) ** 2)
                        ring_distance = math.sqrt((wrist.x - ring_finger_tip.x) ** 2 + (wrist.y - ring_finger_tip.y) ** 2)
                        pinky_distance = math.sqrt((wrist.x - pinky_tip.x) ** 2 + (wrist.y - pinky_tip.y) ** 2)

                        current_time = time.time()
                        if current_time - start_time >= 1:
                            counter += 1
                            data.append((filename,counter, index_distance, middle_distance, ring_distance, pinky_distance))
                            start_time = current_time
            cv2.waitKey(1)
        else:
            break

with open('./src/data/finger_chords.csv', 'w', newline='') as csvfile:
      # Create a CSV writer object
    writer = csv.writer(csvfile)

    for row in data:
        writer.writerow(row)
cap.release()
hands.close()
