import cv2
import mediapipe as mp
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

canvas = None
prev_x, prev_y = 0, 0

brush_thickness = 5
eraser_thickness = 60

points = deque(maxlen=5)

# 🎨 палитра цветов
colors = [
    (0, 255, 0),   # зелёный
    (255, 0, 0),   # синий
    (0, 0, 255),   # красный
    (0, 255, 255), # жёлтый
    (255, 0, 255)  # фиолетовый
]

current_color = colors[0]

def fingers_up(hand):
    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    h, w, _ = frame.shape

    # 🎨 рисуем палитру сверху
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i * 80, 0), ((i + 1) * 80, 60), color, -1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            finger_state = fingers_up(hand)

            lm = hand.landmark[8]
            cx, cy = int(lm.x * w), int(lm.y * h)

            # 🎨 выбор цвета (если палец в зоне палитры)
            if cy < 60:
                index = cx // 80
                if 0 <= index < len(colors):
                    current_color = colors[index]

            # ✊ кулак = очистка
            elif finger_state == [0, 0, 0, 0]:
                canvas = np.zeros_like(frame)
                points.clear()
                prev_x, prev_y = 0, 0

            # 🧽 2 пальца = ластик
            elif finger_state == [1, 1, 0, 0]:
                points.append((cx, cy))

                for i in range(1, len(points)):
                    cv2.line(canvas, points[i - 1], points[i],
                             (0, 0, 0), eraser_thickness)

            # ✏️ 1 палец = рисование
            elif finger_state == [1, 0, 0, 0]:
                points.append((cx, cy))

                for i in range(1, len(points)):
                    cv2.line(canvas, points[i - 1], points[i],
                             current_color, brush_thickness)

            else:
                points.clear()

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.putText(frame, "Palette on top | 1 finger draw | 2 eraser | fist clear",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Virtual Paint", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()