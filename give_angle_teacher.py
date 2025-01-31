import cv2
import csv
import os
import mediapipe as mp
import numpy as np
import streamlit as st

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(landmark1, landmark2, landmark3):
    """Calculate the angle between three pose landmarks."""
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))

    return angle + 360 if angle < 0 else angle

def extract_angles(image):
    """Extract pose angles from an image."""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = [
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value], landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
                calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
            ]
            return angles
    return None

def process_images(images):
    """Processes uploaded images and extracts pose angles."""
    data = []
    for image_file in images:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        angles = extract_angles(image_rgb)
        if angles:
            data.append(angles + [image_file.name])
    return data

# Streamlit UI
st.title("Yoga Pose Angle Extraction")

# File uploader for images
uploaded_files = st.file_uploader("Upload Yoga Pose Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if uploaded_files:
    st.write("Processing images...")
    extracted_data = process_images(uploaded_files)

    if extracted_data:
        # Create CSV content
        csv_filename = "angle_teacher_yoga.csv"
        csv_content = "left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder,left_knee,right_knee,left_ankle,right_ankle,left_hip,right_hip,name_yoga\n"
        for row in extracted_data:
            csv_content += ",".join(map(str, row)) + "\n"

        # Display extracted data as a table
        st.write("Extracted Angles:")
        st.dataframe(extracted_data)

        # Provide download button for CSV file
        st.download_button("Download CSV", csv_content, file_name=csv_filename, mime="text/csv")

