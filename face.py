import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import numpy as np

# Title and description
st.title("Real-Time Face and Eye Detection")
st.markdown(
    """
    - **Live Webcam Feed:** Real-time detection of faces and eyes.
    - **Capture Image Button:** Captures the current frame and displays the detections.
    """
)

# Haar cascade classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Video transformer class
class FaceEyeDetection(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.latest_frame = None  # Store the latest frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Detect eyes within the face ROI
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Store the latest frame for capture
        self.latest_frame = img
        return img

# Stream webcam feed with detection
webrtc_ctx = webrtc_streamer(
    key="face-eye-detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=FaceEyeDetection,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

# Capture image button logic
if st.button("Capture Image"):
    if webrtc_ctx and webrtc_ctx.video_transformer:
        # Retrieve the latest frame
        latest_frame = webrtc_ctx.video_transformer.latest_frame
        if latest_frame is not None:
            # Display the captured frame
            st.image(latest_frame, channels="BGR", caption="Captured Image with Detection")
        else:
            st.warning("No frame available. Please wait for the webcam feed to start.")
    else:
        st.warning("Webcam is not running. Please start the webcam feed.")
