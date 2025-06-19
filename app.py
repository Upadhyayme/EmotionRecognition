import gradio as gr
import cv2
import numpy as np
from keras.models import load_model

# Load model and classifier
model = load_model("emotion_detection_model.h5")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(face)
        emotion = EMOTIONS[np.argmax(prediction)]
        results.append(emotion)
    return ", ".join(results) if results else "No face detected"

interface = gr.Interface(fn=detect_emotion_from_image, inputs="image", outputs="text", title="Emotion Detector")
interface.launch()
