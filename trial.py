import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import io

# Load the MLP model
model_dict = pickle.load(open('./modelMLP.p', 'rb'))
model = model_dict['model']

# Set up MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

# Create a Streamlit app
st.title("Sinovatio Beta - made by Madhu Admuthe")

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Function to run the app
def run_app():
    video_container = st.empty()

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) > 42:
                continue

            prediction = model.predict([np.asarray(data_aux)])
            predicted_proba = model.predict_proba([np.asarray(data_aux)])[0]
            match_percentage = round(max(predicted_proba) * 100, 2)

            if match_percentage < 95.0:
                predicted_character = "Unknown"
                display_text = predicted_character
            else:
                predicted_character = labels_dict[int(prediction[0])]
                display_text = f"{predicted_character} ({match_percentage}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 0, 0), 3, cv2.LINE_AA)

        # Convert OpenCV frame to bytes
        frame_bytes = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[1].tobytes()

        # Display the webcam feed in Streamlit
        video_container.image(frame_bytes, channels="RGB", use_column_width=True)

# Run the app
if __name__ == "__main__":
    run_app()