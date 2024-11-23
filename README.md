# Face_eye_detection
LIVE LINK - https://facepy-8fnit8gjgcxatsypns33gb.streamlit.app/

This application utilizes OpenCV and Streamlit to create a real-time face and eye detection system. Here's a breakdown of its functionality:

1. Libraries and Classifiers:

OpenCV: A powerful library for computer vision tasks, loaded for image processing.
Streamlit: A framework for creating interactive web apps, used for displaying the video feed and results.
Pre-trained Haar cascades: These XML files contain information for detecting faces ("haarcascade_frontalface_default.xml") and eyes ("haarcascade_eye.xml") within images.
2. User Interface:

Streamlit is used to create a user-friendly interface with:
A title: "Real-Time Face and Eye Detection"
Instructions: Explains how to use the "Capture Image" button.
A placeholder: Displays the webcam feed continuously.
A button: "Capture Image" triggers frame capture for analysis.
3. Functionality:

Webcam Access: The application accesses the user's webcam using OpenCV's VideoCapture function.
Image Capture: Clicking the "Capture Image" button captures the current frame from the webcam feed.
Image Processing:
The captured image is converted to grayscale for improved detection accuracy.
Face detection is performed using the face_classifier on the grayscale image. Detected faces are marked with red rectangles and labeled "Face".
For each detected face, a region of interest (ROI) is defined.
Eye detection is performed within the face ROI using the eye_classifier. Detected eyes are marked with green rectangles and labeled "Eye".
Image Display:
The captured image with detected faces and eyes is converted to RGB format for displaying in Streamlit.
The processed image is displayed using Streamlit's image function.
Webcam Feed: While the "Capture Image" button isn't clicked, the live webcam feed is displayed continuously.
4. Release and Cleanup:

After image capture or when the application closes, the webcam is released using cap.release() to free system resources.
Real-World Applications:

This code serves as a basic building block for various real-world applications, including:

Security Systems: Facial recognition systems can be built upon this foundation, potentially identifying authorized personnel.
Drowsiness Detection: Monitoring eye closure patterns can help detect driver drowsiness.
Accessibility Tools: Facial detection can be used to focus user interfaces or adjust settings based on a user's position.
Interactive Displays: Eye tracking can be implemented for interactive displays, allowing control based on user gaze.
Image/Video Annotation: This code can be adapted to automate the process of marking faces and eyes in images or video frames for training data creation.
