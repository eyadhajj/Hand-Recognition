import cv2
from PIL import Image
import random as rd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

#===================================================================#
# VARIABLES
#===================================================================#
hand_image_path = "hands/hands3.jpg"
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
LANDMARK_NUM_COLOR = (255, 0, 0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing_live = mp.solutions.drawing_utils

#===================================================================#
# GESTURE RECOGNIZER SETUP (no TensorFlow needed)
#===================================================================#
gesture_base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
gesture_options = vision.GestureRecognizerOptions(
    base_options=gesture_base_options,
    num_hands=4
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

#===================================================================#
# BRIGHTNESS
#===================================================================#
def determine_brightness():
    pass

#===================================================================#
# BOX DRAWING AROUND HAND
#===================================================================#
def draw_border(image, hand_landmarks):
    width = image.shape[1]
    height = image.shape[0]
    x_cords = []
    y_cords = []
    border_spacing = 40
    for landmark in hand_landmarks.landmark:
        x_cords.append(int(landmark.x * width))
        y_cords.append(int(landmark.y * height))
    min_x_point = min(x_cords)
    min_y_point = min(y_cords)
    max_x_point = max(x_cords)
    max_y_point = max(y_cords)
    start_point = (min_x_point - border_spacing, min_y_point - border_spacing)
    end_point = (max_x_point + border_spacing, max_y_point + border_spacing)
    cv2.rectangle(image, start_point, end_point, (255, 255, 255), 3)

#===================================================================#
# DISTANCE CALCULATION FUNCTION
#===================================================================#
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

stream = cv2.VideoCapture(0)
#===================================================================#

while stream.isOpened():
    ret, frame = stream.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    #===================================================================#
    # GESTURE RECOGNITION using MediaPipe GestureRecognizer
    #===================================================================#
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    gesture_result = gesture_recognizer.recognize(mp_image)

    # Build a dict of gesture per hand index for easy lookup
    gesture_per_hand = {}
    if gesture_result.gestures:
        for i, gestures in enumerate(gesture_result.gestures):
            if gestures:
                top = gestures[0]
                gesture_per_hand[i] = (top.category_name, top.score)
    #===================================================================#

    if results.multi_hand_landmarks:

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            height, width, _ = frame.shape

            # Drawing corner points
            top_left = (10, 10)
            top_right = (frame.shape[1] - 11, 10)
            bottom_left = (10, frame.shape[0] - 11)
            bottom_right = (frame.shape[1] - 11, frame.shape[0] - 11)

            cv2.circle(frame, top_left, 10, (0, 0, 255), -1)
            cv2.circle(frame, top_right, 10, (0, 0, 255), -1)
            cv2.circle(frame, bottom_left, 10, (0, 0, 255), -1)
            cv2.circle(frame, bottom_right, 10, (0, 0, 255), -1)

            mp_drawing_live.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for connection in solutions.hands.HAND_CONNECTIONS:
                points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark]
                color = (255, 255, 255)
                start_idx, end_idx = connection
                cv2.line(frame, points[start_idx], points[end_idx], color, 2)

            #===================================================================#
            # DISPLAY GESTURE LABEL
            #===================================================================#
            if hand_idx in gesture_per_hand:
                class_name, confidence = gesture_per_hand[hand_idx]
                display_name = class_name if confidence > 0.7 else "Uncertain Gesture"
            else:
                display_name = "Uncertain Gesture"

            label_y = 50 + hand_idx * 40  # offset label per hand so they don't overlap
            cv2.putText(frame, display_name, (10, label_y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #===================================================================#

        cv2.imshow('Live Hand Recognition', frame)

        if cv2.waitKey(1) == ord('a'):
            break

stream.release()
cv2.destroyAllWindows()