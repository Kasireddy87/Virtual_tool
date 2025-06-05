import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Drawing settings
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
color_names = ['Blue', 'Green', 'Red', 'White']
draw_color = colors[0]
brush_thickness = 8
eraser_thickness = 20

# Canvas and previous coordinates
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Detect which fingers are up
def fingers_up(lm):
    fingers = []
    fingers.append(1 if lm[4].x < lm[3].x else 0)  # Thumb
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)
            fingers = fingers_up(lm)

            # ✋ Color Selection: 3 fingers up (index, middle, ring)
            if fingers == [0, 1, 1, 1, 0] and cy < 100:
                section = cx // 320
                if section < len(colors):
                    draw_color = colors[section]
                xp, yp = 0, 0

            # ✌️ Draw: Only index & middle finger up
            elif fingers == [0, 1, 1, 0, 0]:
                cv2.circle(img, (cx, cy), 10, draw_color, -1)
                if xp == 0 and yp == 0:
                    xp, yp = cx, cy
                cv2.line(img, (xp, yp), (cx, cy), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (cx, cy), draw_color, brush_thickness)
                xp, yp = cx, cy

            # ☝️ Erase: Only index finger up
            elif fingers == [0, 1, 0, 0, 0]:
                cv2.circle(img, (cx, cy), 10, (0, 0, 0), -1)
                if xp == 0 and yp == 0:
                    xp, yp = cx, cy
                cv2.line(img, (xp, yp), (cx, cy), (0, 0, 0), eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (cx, cy), (0, 0, 0), eraser_thickness)
                xp, yp = cx, cy

            # 🤏 Resize eraser: Thumb + index finger only
            elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
                thumb_tip = lm[4]
                index_tip = lm[8]
                dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                eraser_thickness = int(20 + (1 - dist) * 80)
                eraser_thickness = max(10, min(eraser_thickness, 80))
                xp, yp = 0, 0

            # ✋ All fingers up: vertical hand movement to resize
            elif fingers == [1, 1, 1, 1, 1]:
                index_y = int(lm[8].y * h)
                eraser_thickness = int(np.interp(index_y, [100, 600], [10, 80]))
                xp, yp = 0, 0

            else:
                xp, yp = 0, 0

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay canvas on image
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # 🎨 Draw color palette
    for i, color in enumerate(colors):
        cv2.rectangle(img, (i * 320, 0), ((i + 1) * 320, 100), color, -1)
        cv2.putText(img, color_names[i], (i * 320 + 100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # ✅ Show selected color
    cv2.putText(img, f'Selected Color: {color_names[colors.index(draw_color)]}',
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, draw_color, 3)

    # 📏 Eraser thickness slider (left)
    cv2.rectangle(img, (50, 100), (80, 600), (200, 200, 200), 2)
    slider_pos = int(np.interp(eraser_thickness, [10, 80], [600, 100]))
    cv2.circle(img, (65, slider_pos), 10, (0, 0, 255), -1)
    cv2.putText(img, f'{eraser_thickness}', (40, slider_pos - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show instructions on screen
    cv2.putText(img, "Draw: Index & Middle | Erase: Index Only | Resize: All Fingers | Color: 3 Fingers", 
                (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show output
    cv2.imshow("Gesture Whiteboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
