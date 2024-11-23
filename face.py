import cv2
import streamlit as st
import numpy as np

# Load Haar cascades for face and eye detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Streamlit app title
st.title('Real-Time Face and Eye Detection')

# Function to detect faces and eyes in a given image
def detect_faces_and_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Region of interest (ROI) for eyes within the detected face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        # Detect eyes
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Camera input using Streamlit
camera_input = st.camera_input("Capture Image")

# If a picture is taken
if camera_input:
    # Read the image
    img = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), 1)

    # Perform face and eye detection
    detect_faces_and_eyes(img)

    # Convert image for display in Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show the processed image
    st.image(img_rgb, channels="RGB", caption="Captured Image with Face and Eye Detection")

# Add some instructions
st.write("""
    **Instructions**:
    1. Use the camera input above to capture your image.
    2. Once the image is captured, face and eye detection will be performed.
""")
