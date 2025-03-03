<h1>Hand Gesture Recognition</h1>

With the advancement of computer vision, we are able to harness the ability to communicate with machines to recognise our gestures in order to trigger an action or to signal communication. As of now, my project can predict gestures and display it on the screen.

<h2>Introduction</h2>
This project uses Google's MediaPipe, an open-source framework, to detect hand landmarks in real-time. By integrating machine learning, the application recognizes hand gestures and maps them to specific actions. Additionally, OpenCV is used to process images and video streams efficiently, providing an interactive interface.

<h2>How does it work?</h2>

<h3>Video Capture</h3>
OpenCV will be responsible for live video capturing and drawing. Each frame is mirrored for a better experience and coverts the frames to RGB. We are required to convert these frames to RGB so it can work with MediaPipe, as each frame is initially BGR.

<h3>Hand Detection and Landmark Extraction</h3>
MediaPipe's palm detection model detects hands within a capture (can be configured to detect multiple hands if wanted). Each hand has 21 key points and are all extracted, mainly for ML purposes. Each given landmark are stored as coordinated (x, y, z). These are stored in <code>hand_landmarks.landmark</code> for our functions to process and use for further development. 

<h3>Gesture Recognition</h3>
For the model input, the coordinates of all the landmark locations are converted into a Numpy array for the model to use and predict the gesture. The confidence level is set and if the gesture is not above that threshold then it will display "uncertain gesture".

<h2>Modules/Libraries used</h2>
- Numpy -> Used for complex mathematical computations and array handling. <br>
- OpenCV -> Image/Video processing. <br>
- MediaPipe -> Live hand landmark detection and tracking. <br>
- PIL -> Primarily used for image processing. <br>
- Keras -> Used for loading and utilisation of pretrained models. <br>
