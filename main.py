import cv2
from PIL import Image
import random as rd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from keras.models import load_model

hand_image_path = "hands/hands3.jpg"


try:
    hand_image = Image.open(hand_image_path)
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
    exit()

def convert_image_mp(image):
    image_np = np.array(image, dtype=np.uint8)  # Convert PIL to NumPy (uint8 format)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

def load_image(image_path):
    pil_image = Image.open(image_path)
    return pil_image

def mp_input_image(image_path):
    image = mp.Image.create_from_file(image_path)
    return image

image = mp_input_image(hand_image_path)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=4)

detector = vision.HandLandmarker.create_from_options(options)

detection_result = detector.detect(image)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
LANDMARK_NUM_COLOR = (255, 0, 0) 

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    base_palm = [5, 9, 13, 17]

    # Debugging: Check if hands are detected
    print(f"Detected {len(hand_landmarks_list)} hands")

    if len(hand_landmarks_list) == 0:
        print("No hands detected! Check image and lighting.")
        return annotated_image  # Return original image if no hands are found

    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        handedness = handedness_list[idx]

        # Draw hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get image dimensions
        height, width, _ = annotated_image.shape

        # Draw handedness label (Left or Right hand)
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Display landmark numbers
        for num, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if num not in base_palm:
                color = (0, 0, 255)  # Red for fingertips
                offset_x, offset_y = -30, 10
            else:
                color = (255, 0, 0)  # Blue for palm
                offset_x, offset_y = -40, -10

            cv2.putText(annotated_image, str(num), (x + offset_x, y + offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return annotated_image  

def draw_landmarks_blank(detction_result, image_size=(500,500), bg_colour=(0, 0, 0)):
    blank_image = np.full((image_size[1], image_size[0], 3), bg_colour, dtype=np.uint8)
    
    hand_landmarks_list = detection_result.hand_landmarks
    
    if len(hand_landmarks_list) == 0:
        print("No hands detected! Returning blank image.")
        return blank_image  # blank image if no hands are detected

    for hand_landmarks in hand_landmarks_list:
        # normalized coordinates to pixel coordinates
        points = [(int(landmark.x * image_size[0]), int(landmark.y * image_size[1])) for landmark in hand_landmarks]

        for idx, (x, y) in enumerate(points):
            cv2.circle(blank_image, (x, y), 5, (0, 255, 0), -1)  # Green circles for landmarks
            cv2.putText(blank_image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # connections, will edit to better liking 
        for connection in solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(blank_image, points[start_idx], points[end_idx], (255, 0, 0), 2)  # Blue lines for connections

    return blank_image


print(f"Image shape: {image.numpy_view().shape}")

image_np = image.numpy_view()
if image_np.shape[-1] == 4:  # If the image has an alpha channel (RGBA)
    print(f"Converting image from RGBA to RGB.")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)


#Uncomment for hand recgonition on an image
"""""
annotated_image = draw_landmarks_on_image(image_np, detection_result)

cv2.imshow("Hand Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Uncomment for hand recognition display on an empty canvas 
"""
landmark_only_image = draw_landmarks_blank(detection_result, image_size=(500, 500))

cv2.imshow("Landmark-Only Display", landmark_only_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


# Live stream hand recognition

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing_live = mp.solutions.drawing_utils

gesture_model = load_model('mp_hand_gesture')

# loading class names
f = open('gesture.names', 'r')
class_names = f.read().split('\n')
f.close()
print(class_names)


stream = cv2.VideoCapture(0)

while stream.isOpened():
    ret , frame = stream.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:


        for hand_landmarks in results.multi_hand_landmarks:

            height, width, _ = frame.shape

            mp_drawing_live.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for connection in solutions.hands.HAND_CONNECTIONS:
                
               
                points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark]
                color = (88, 205, 54)  # Random RGB color

                start_idx, end_idx = connection
                cv2.line(frame, points[start_idx], points[end_idx], color, 2)  # Blue lines for connections
                
            landmarks = []
            for landmark in hand_landmarks.landmark:
                lmx = (landmark.x * width)
                lmy = (landmark.y * height)
                
                landmarks.append([lmx, lmy])
            # Convert to a NumPy array and reshape for prediction (1, number_of_features)
            
            prediction = gesture_model.predict([landmarks])
            print("Prediction:", prediction)

            classID = np.argmax(prediction)
            classID = int(classID)
            print(f"Predicted classID: {classID}")
 
            if classID >= 0 and classID < len(class_names):
                class_name = class_names[classID]
            else:
                class_name = "Unknown Gesture"
                print("Error: classID is out of range.")

        cv2.putText(frame, class_name, (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Live Hand Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
stream.release()
cv2.destroyAllWindows()