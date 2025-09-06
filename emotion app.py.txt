import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load pre-trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit page setup
st.set_page_config(page_title="Emotion Detection", page_icon="😎", layout="centered")
st.title("😎 Real-Time Emotion Detection")
st.markdown("### Upload an image or start your webcam to detect emotions!")

# Upload image option
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion detection function
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[prediction.argmax()]
        confidence = prediction.max()

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence*100:.2f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

# If image uploaded
if uploaded_image:
    img = Image.open(uploaded_image)
    img = np.array(img)
    result_img = detect_emotion(img)
    st.image(result_img, channels="BGR", caption="Detected Emotion")

# Live webcam detection
if st.button("Start Webcam"):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_img = detect_emotion(frame)
        stframe.image(result_img, channels="BGR")
    cap.release()
