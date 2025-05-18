import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

# Load model and label binarizer
model = load_model("models/mask_detector.h5")
lb = pickle.load(open("models/label_binarizer.pickle", "rb"))

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        pred = model.predict(face)[0][0]
        label = "Mask 😷" if pred < 0.5 else "No Mask ❌"
        color = (0, 255, 0) if label == "Mask 😷" else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cv2.imshow("Mask Detection - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
