import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

try:
    hand_image = Image.open("hands/hand1.jpg")  # Open the image using PIL
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
    exit()

# Function to convert PIL image to Mediapipe Image
def convert_image_mp(image):
    image_np = np.array(image, dtype=np.uint8)  # Convert PIL to NumPy (uint8 format)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

# Function to draw landmarks on image (using OpenCV for display)
def draw_landmarks(image, detection_result):
    # Convert PIL to NumPy array (this ensures `shape` is available for drawing)
    image_np = np.array(image)
    for hand_landmarks in detection_result:
        # Ensure we're passing the correct landmarks to the drawing function
        mp_drawing.draw_landmarks(
            image_np,  # Image to draw landmarks on
            hand_landmarks,  # Detected hand landmarks (as NormalizedLandmarkList)
            mp_hands.HAND_CONNECTIONS  # Connections between landmarks
        )
    return image_np

# Mediapipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the model using mediapipe.tasks
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

# Convert the image to mp.Image format for task-based hand detection
image = convert_image_mp(hand_image)

# Create the hand landmark detector
detector = vision.HandLandmarker.create_from_options(options)

# Run hand landmark detection
detection_result = detector.detect(image)

# Ensure we have detected hands and draw the landmarks if present
if detection_result.hand_landmarks:
    # Draw landmarks and get a NumPy array
    image_with_landmarks = draw_landmarks(hand_image, detection_result.hand_landmarks)

    # Convert from RGB to BGR for OpenCV (since OpenCV expects BGR)
    image_with_landmarks_bgr = cv2.cvtColor(image_with_landmarks, cv2.COLOR_RGB2BGR)

    # Show the image with landmarks using OpenCV
    cv2.imshow("Hand Landmarks", image_with_landmarks_bgr)
    cv2.waitKey(0)

# Clean up and close the OpenCV window
cv2.destroyAllWindows()
