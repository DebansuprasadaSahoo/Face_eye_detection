import cv2
import streamlit as st
import numpy as np

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Streamlit app title
st.title('Real-Time Face and Eye Detection')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to detect faces and eyes in a given image
def detect_faces_and_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Streamlit UI
frame_placeholder = st.empty()
capture_button = st.button('Capture Image')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and eyes
    detect_faces_and_eyes(frame)

    # Display the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels='RGB')

    if capture_button:
        # Capture the current frame and display it
        st.image(frame_rgb, channels='RGB')
        break

# Release the camera
cap.release()
