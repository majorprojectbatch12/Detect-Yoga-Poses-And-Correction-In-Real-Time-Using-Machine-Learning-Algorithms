import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

# Initialize Mediapipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # For drawing keypoints

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

def extract_pose_angles(results):
    """Extracts pose angles from Mediapipe results."""
    angles = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE)
        ]

        for p1, p2, p3 in keypoints:
            angles.append(calculate_angle(landmarks[p1.value], landmarks[p2.value], landmarks[p3.value]))

    return angles

def predict(img_path, model):
    """Predicts pose from an image and returns processed frame."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    results = pose.process(img)

    if results.pose_landmarks:
        list_angles = extract_pose_angles(results)
        y = model.predict([list_angles])[0]

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(img, str(y), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (215, 215, 0), 3)

    return img, y  # Returning image and prediction label

def predict_video(model, video_path):
    """Predicts poses in a video and returns frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            list_angles = extract_pose_angles(results)
            y = model.predict([list_angles])[0]

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, str(y), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        frames.append(img)

    cap.release()
    return frames  # Return a list of processed frames

def evaluate(data_test, model):
    """Evaluates model performance and returns confusion matrix + classification report."""
    target = data_test['target'].values.tolist()
    predictions = [model.predict([data_test.iloc[i, :-1].values.tolist()])[0] for i in range(len(data_test))]

    cm = confusion_matrix(target, predictions)
    report = classification_report(target, predictions, output_dict=True)

    return cm, report
