import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Title and Instructions
st.title("Real-Time Face and Eye Detection")
st.markdown(
    """
    - **Start Webcam:** Press the button below to initialize the webcam feed.
    - **Capture Feature:** Real-time face and eye detection overlayed on your webcam feed.
    """
)

# Load Haar Cascade Classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Define Video Transformer Class
class FaceEyeDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array in BGR format
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        for (x, y, w, h) in faces:
            # Draw rectangle around faces
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Eye Detection within Face Region
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# Start Webcam Feed
try:
    webrtc_ctx = webrtc_streamer(
        key="face-eye-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceEyeDetectionTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Enable video; disable audio
        async_processing=True,  # Enable asynchronous video processing
    )

    # Status Check for Webcam
    if webrtc_ctx.state.playing:
        st.success("Webcam is running successfully!")
    else:
        st.warning("Webcam is not running. Please allow access to your camera.")
except Exception as e:
    st.error(f"Error initializing webcam: {e}")
