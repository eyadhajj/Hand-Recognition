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


# Variables 
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
gesture_model = load_model('mp_hand_gesture')


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
detection_result = detector.detect(image)

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

    for hand_landmarks in hand_landmarks_list: # goes through all normalized coordinates of hand lanmarks

        # normalized coordinates to pixel coordinates
        points = [(int(landmark.x * image_size[0]), int(landmark.y * image_size[1])) for landmark in hand_landmarks] #21 landmarks based off doumentation.

        for idx, (x, y) in enumerate(points): # 21 landmarks
            cv2.circle(blank_image, (x, y), 5, (0, 255, 0), -1)  # draws a green dot, at those coordinates, 5 pixel radius and -1 to represent filled
            cv2.putText(blank_image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # add label text for landmark 

        # connections, will edit to better liking 
        for connection in solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection # a start point to an endpoint is a connection
            cv2.line(blank_image, points[start_idx], points[end_idx], (255, 0, 0), 2)  # draw line for those connections
 
    return blank_image

print(f"Image shape: {image.numpy_view().shape}")

image_np = image.numpy_view()
if image_np.shape[-1] == 4:  # If the image has an alpha channel (RGBA)
    print(f"Converting image from RGBA to RGB.")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

annotated_image = draw_landmarks_on_image(image_np, detection_result)
cv2.imshow("Hand Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
