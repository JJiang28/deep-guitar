import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture('./test_video.mp4')
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils
p_time = 0
# out = cv2.VideoWriter('output.avi', -1, 20.0, size)

while True:
    # c_time = time.time()
    # fps = 1 / (c_time - p_time)
    # p_time = c_time

    success, image = cap.read()
    if success:
        # cv2.putText(image, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                print(handLms.landmark[0])
                # avg_center_x, avg_center_y = 0, 0

                # for idx, lm in enumerate(handLms.landmark):
                #     if idx in self.center_tracking_landmarks:
                #         h, w, c = image.shape
                #         cx, cy = int(lm.x * w), int(lm.y * h)
                #         avg_center_x += cx
                #         avg_center_y += cy

                # avg_center_x = avg_center_x // len(center_tracking_landmarks)
                # avg_center_y = avg_center_y // len(center_tracking_landmarks)
                # cv2.circle(image, (avg_center_x, avg_center_y), 25, (255, 0, 255), cv2.FILLED)

                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

        # out.write(image)

        cv2.imshow("cv2", image)
        cv2.waitKey(1)
    else:
        break

cap.release()