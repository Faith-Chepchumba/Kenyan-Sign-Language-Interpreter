# sign_language_recognition.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('sign_language_model.h5')

# Load webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the frame for model input
    resized_frame = cv2.resize(gray, (64, 64))
    reshaped_frame = resized_frame.reshape(1, 64, 64, 1)
    
    # Predict the gesture
    prediction = model.predict(reshaped_frame)
    predicted_label = np.argmax(prediction)
    
    # Display the prediction
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Sign Language Interpreter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()