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


#===================================================================#
# VARIABLES
#===================================================================#hand_image_path = "hands/hands3.jpg"
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
gesture_model = load_model('mp_hand_gesture') # Load model from keras -> pretrained neural network


#===================================================================#
# LOADING CLASS NAMES FOR GESTURE PREDICTIONS
#===================================================================#
f = open('gesture.names', 'r')
class_names = f.read().split('\n')
f.close()
#===================================================================#

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
    height =  image.shape[0]
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

#===================================================================#
# DISTANCE CALCULATION FUNCTION
#===================================================================#
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

stream = cv2.VideoCapture(0)
#===================================================================#

while stream.isOpened():
    ret , frame = stream.read()


    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB, for mediapipe 

    results = hands.process(frame_rgb) # use the hand model to process the frame

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            height, width, _ = frame.shape

            # Drawing points on the screen
            # This is going to be used to extract pixel colours so that I can make an algorithm to change connection color based off background
            top_left = (10, 10)
            top_right = (frame.shape[1] - 11, 10)
            bottom_left = (10, frame.shape[0] - 11)
            bottom_right = (frame.shape[1] - 11, frame.shape[0] - 11)

            cv2.circle(frame, top_left, 10, (0,0,255), -1)
            cv2.circle(frame, top_right, 10, (0,0,255), -1)
            cv2.circle(frame, bottom_left, 10, (0,0,255), -1)
            cv2.circle(frame, bottom_right, 10, (0,0,255), -1)

            mp_drawing_live.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # drawing the landmarks
            
            for connection in solutions.hands.HAND_CONNECTIONS:
                
                points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark]
                color = (255, 255, 255)  # Random RGB color

                start_idx, end_idx = connection
                cv2.line(frame, points[start_idx], points[end_idx], color, 2) 
                
            landmarks = []
            for landmark in hand_landmarks.landmark:
                lmx = (landmark.x * width)
                lmy = (landmark.y * height)
                
                landmarks.append([lmx, lmy])

#===================================================================#
# MACHINE LEARNING GESTURE PREDICTIONS
#===================================================================#

            input_data = np.array(landmarks).reshape(1, 21, 2) # numpy array layouts the landmarks, 2 hands == 41 landmarks with 2 coords each
            # batch size = 1, to process 1 hands landmark at a time, if i do 3 then it processes 3 sepate hands in parallel 
            # 21 landmarks
            # coords per landmark (x, y)


            prediction = gesture_model.predict(input_data) # using the input data, predict the gesture given from the model
            confidence = np.max(prediction) # how confident the prediction is
            classID = np.argmax(prediction) # name of gesure predicted

            if confidence > 0.7:  # Confidence threshold
                class_name = class_names[classID] #  display the name of the predecicted gesture
            else:
                class_name = "Uncertain Gesture" # if not detected, display unknown

            cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#===================================================================#


 # Distance between finger and thumb tracker.
        """
        # To calculate distance between thumb tip and index finger tip -- to be used for gesture action
        #     thumb_tip = hand_landmarks.landmark[4]
        #     index_tip = hand_landmarks.landmark[8]

        #     # Coordinates of thumb
        #     x1, y1 = int(thumb_tip.x * width), int(thumb_tip.y * height)

        #     # Coordinates of index finger
        #     x2, y2 = int(index_tip.x * width), int(index_tip.y * height)

        #     distance = calculate_distance(x1, y1, x2, y2) # distance calcuated (in pixels)

        #     cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw a line connecting between thumb and index coordinates

        #     # Display distance on image
        #     cv2.putText(frame, f'{distance:.2f} px', (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 

        # draw_border(frame, hand_landmarks)
        """
        cv2.imshow('Live Hand Recognition', frame)

    # Exit when 'a' is pressed
        if cv2.waitKey(1) == ord('a'):
            break

stream.release()
cv2.destroyAllWindows()