import streamlit as st
import cv2
import numpy as np
import random
import time
from PIL import Image

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Create a Streamlit app
st.title("NanoScan - Eye Tracking")

# Open webcam using OpenCV
cap = cv2.VideoCapture(0)

dot_x, dot_y = random.randint(100, 500), random.randint(100, 400)
last_dot_time = time.time()

start_time = time.time()
duration = 60  # seconds

tracking_errors = []
max_reasonable_dist = 200  # Pixels — anything beyond this is likely bad detection

st.write("NanoScan running for 60 seconds... Follow the red dot with your eyes.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed = current_time - start_time

    if elapsed > duration:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eye_centers = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes[:2]:
            center = (x + ex + ew//2, y + ey + eh//2)
            eye_centers.append(center)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

        if len(eye_centers) == 2:
            break  # Use only first face with both eyes

    # Draw red target dot
    cv2.circle(frame, (dot_x, dot_y), 10, (0, 0, 255), -1)

    # Move dot every 3 seconds
    if current_time - last_dot_time > 3:
        dot_x, dot_y = random.randint(100, 500), random.randint(100, 400)
        last_dot_time = current_time

    # Use average eye position
    if len(eye_centers) == 2:
        avg_eye_x = (eye_centers[0][0] + eye_centers[1][0]) // 2
        avg_eye_y = (eye_centers[0][1] + eye_centers[1][1]) // 2
        eye_center = (avg_eye_x, avg_eye_y)

        dist = np.linalg.norm(np.array([dot_x, dot_y]) - np.array(eye_center))

        if dist < max_reasonable_dist:
            tracking_errors.append(dist)  # Only keep reasonable values

        cv2.putText(frame, f"Error: {int(dist)} px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Time Left: {int(duration - elapsed)}s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Convert OpenCV image (BGR) to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the image in Streamlit
    st.image(frame_rgb, channels="RGB", use_column_width=True)

    # Break if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# --- Score ---
if tracking_errors:
    avg_error = np.mean(tracking_errors)
    norm_error = min(avg_error / max_reasonable_dist, 1.0)
    score = int((1 - norm_error**1.5) * 100)  # Slight curve to reward good tracking
else:
    avg_error = None
    score = 0

# Display the result in Streamlit
st.write("### NanoScan Report")
st.write(f"Eye Tracking Score: {score}/100")
st.write(f"Average Tracking Error: {int(avg_error) if avg_error else 'N/A'} px")
st.write(f"Frames Tracked: {len(tracking_errors)}")
