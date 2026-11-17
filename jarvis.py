
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import os

# ==========================================
# JARVIS V1.1 - HAND GESTURE CONTROLLER
# ==========================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03

SCREEN_W, SCREEN_H = pyautogui.size()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "hand_landmarker.task is missing from JARVIS folder!"
    )

# ==========================================
# MEDIAPIPE TASKS SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)

# ==========================================
# CAMERA SETUP
# ==========================================

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    landmarker.close()
    raise RuntimeError("Webcam not detected!")

# ==========================================
# VARIABLES
# ==========================================

CONTROL_ENABLED = False

SMOOTHING = 5

previous_x = SCREEN_W // 2
previous_y = SCREEN_H // 2

last_click = 0
last_right_click = 0

previous_pinch = False
previous_right_gesture = False

previous_scroll_y = None

start_time = time.perf_counter()
last_timestamp = -1

# Hand connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def get_fingers(lm):
    return [
        lm[8].y < lm[6].y,     # Index
        lm[12].y < lm[10].y,   # Middle
        lm[16].y < lm[14].y,   # Ring
        lm[20].y < lm[18].y    # Pinky
    ]


def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    points = []

    for point in landmarks:
        x = int(point.x * w)
        y = int(point.y * h)
        points.append((x, y))

    for start, end in CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )


def show_text(frame, message, y, color=(0, 255, 0)):
    cv2.putText(
        frame,
        message,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

print("================================")
print("       JARVIS V1.1 STARTED")
print("================================")
print("SPACE = Activate / Pause")
print("Q = Quit")
print("Emergency: Move cursor to top-left corner")

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame error!")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            (time.perf_counter() - start_time) * 1000
        )

        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        results = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # ==================================
        # HAND DETECTION
        # ==================================

        if results.hand_landmarks:

            lm = results.hand_landmarks[0]

            draw_hand(frame, lm)

            fingers = get_fingers(lm)

            index = lm[8]
            thumb = lm[4]

            pinch = distance(index, thumb) < 0.045

            # ==================================
            # CONTROL ONLY WHEN ACTIVATED
            # ==================================

            if CONTROL_ENABLED:

                # LEFT CLICK - PINCH
                if pinch and not previous_pinch:

                    if time.time() - last_click > 0.5:

                        pyautogui.click()
                        last_click = time.time()

                        show_text(
                            frame,
                            "LEFT CLICK",
                            100
                        )

                previous_pinch = pinch

                # RIGHT CLICK - TWO FINGERS
                right_gesture = (
                    fingers[0]
                    and fingers[1]
                    and not fingers[2]
                    and not fingers[3]
                    and not pinch
                )

                if right_gesture and not previous_right_gesture:

                    if time.time() - last_right_click > 0.7:

                        pyautogui.rightClick()
                        last_right_click = time.time()

                        show_text(
                            frame,
                            "RIGHT CLICK",
                            130
                        )

                previous_right_gesture = right_gesture

                # SCROLL - THREE FINGERS
                scroll_gesture = (
                    fingers[0]
                    and fingers[1]
                    and fingers[2]
                    and not fingers[3]
                    and not pinch
                )

                if scroll_gesture:

                    center_y = (
                        lm[8].y +
                        lm[12].y +
                        lm[16].y
                    ) / 3

                    if previous_scroll_y is not None:

                        movement = center_y - previous_scroll_y

                        if movement < -0.018:
                            pyautogui.scroll(2)
                        elif movement > 0.018:
                            pyautogui.scroll(-2)

                    previous_scroll_y = center_y

                    show_text(
                        frame,
                        "SCROLL MODE",
                        160
                    )

                else:
                    previous_scroll_y = None

                # MOUSE MOVEMENT - INDEX ONLY
                if (
                    fingers[0]
                    and not fingers[1]
                    and not fingers[2]
                    and not fingers[3]
                    and not pinch
                ):

                    target_x = np.interp(
                        index.x,
                        [0.1, 0.9],
                        [0, SCREEN_W]
                    )

                    target_y = np.interp(
                        index.y,
                        [0.1, 0.9],
                        [0, SCREEN_H]
                    )

                    current_x = (
                        previous_x +
                        (target_x - previous_x) / SMOOTHING
                    )

                    current_y = (
                        previous_y +
                        (target_y - previous_y) / SMOOTHING
                    )

                    pyautogui.moveTo(
                        int(current_x),
                        int(current_y)
                    )

                    previous_x = current_x
                    previous_y = current_y

                    show_text(
                        frame,
                        "MOUSE CONTROL",
                        190
                    )

            else:
                previous_pinch = False
                previous_right_gesture = False
                previous_scroll_y = None

        else:
            previous_pinch = False
            previous_right_gesture = False
            previous_scroll_y = None

        # ==================================
        # DISPLAY STATUS
        # ==================================

        if CONTROL_ENABLED:
            show_text(
                frame,
                "JARVIS: ACTIVE",
                35,
                (0, 255, 0)
            )
        else:
            show_text(
                frame,
                "JARVIS: PAUSED",
                35,
                (0, 0, 255)
            )

        show_text(
            frame,
            "SPACE: TOGGLE | Q: EXIT",
            frame.shape[0] - 20
        )

        cv2.imshow(
            "JARVIS - HAND GESTURE CONTROL",
            frame
        )

        # ==================================
        # KEYBOARD CONTROLS
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 32:

            CONTROL_ENABLED = not CONTROL_ENABLED

            previous_pinch = False
            previous_right_gesture = False
            previous_scroll_y = None

            print(
                "JARVIS ACTIVE"
                if CONTROL_ENABLED
                else "JARVIS PAUSED"
            )

except KeyboardInterrupt:
    print("JARVIS stopped by user")

except Exception as error:
    print("JARVIS ERROR:", error)

finally:

    camera.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("JARVIS SYSTEM CLOSED")
