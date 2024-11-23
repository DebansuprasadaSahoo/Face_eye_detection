import cv2
import streamlit as st
import numpy as np

# Load Haar cascades for face and eye detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Streamlit setup
st.title("Real-Time Face and Eye Detection")
st.markdown("Click the 'Capture Image' button to capture a photo and detect faces and eyes!")

# Streamlit camera feed
cap = cv2.VideoCapture(1)

# Placeholder for showing webcam feed continuously
frame_placeholder = st.empty()

# Button to capture the image
capture_button = st.button("Capture Image")

if capture_button:
    # Capture the current frame when the button is clicked
    ret, img = cap.read()
    if not ret:
        st.write("Failed to capture image!")
    else:
        # Convert the image to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Region of interest (ROI) for eyes within the detected face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect eyes within the face ROI
            eyes = eye_classifier.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the image to RGB for display in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the captured image with detections
        st.image(img_rgb, channels="RGB", use_column_width=True)

    # Release the webcam after the capture
    cap.release()

else:
    # If the capture button isn't clicked, keep showing the webcam feed
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        # Convert to RGB and display the live feed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

# Cleanup and release the camera after use
cap.release()
