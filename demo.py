import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import streamlit as st

# Create a pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # For drawing keypoints

# Function to calculate angle between three points
def calculate_angle(landmark1, landmark2, landmark3, select=''):
    if select == '1':
        x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
        x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
        x3, y3, _ = landmark3.x, landmark3.y, landmark3.z
        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    else:
        radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0] - landmark2[0]) - np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
        angle = np.abs(np.degrees(radians))

    return angle + 360 if angle < 0 else angle

def correct_feedback(model, video_path, input_csv):
    """
    Processes the video frame by frame, applies pose detection, and returns frames with feedback.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return []

    # Load accurate angles for reference
    accurate_angle_lists = []
    angle_name_list = ["L-wrist", "R-wrist", "L-elbow", "R-elbow", "L-shoulder", "R-shoulder",
                       "L-knee", "R-knee", "L-ankle", "R-ankle", "L-hip", "R-hip"]
    
    angle_coordinates = [
        [13, 15, 19], [14, 16, 18], [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], 
        [23, 25, 27], [24, 26, 28], [23, 27, 31], [24, 28, 32], [24, 23, 25], [23, 24, 26]
    ]
    
    correction_value = 30  # Allowed deviation
    fps_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = [calculate_angle(landmarks[angle_coordinates[i][0]],
                                      landmarks[angle_coordinates[i][1]],
                                      landmarks[angle_coordinates[i][2]], '1') for i in range(12)]

            # Make a prediction
            y = model.predict([angles])
            predicted_pose = str(y[0])

            # Display predicted class
            cv2.putText(frame, predicted_pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            correct_angle_count = 0
            for i in range(12):
                point_a = (int(landmarks[angle_coordinates[i][0]].x * frame.shape[1]),
                           int(landmarks[angle_coordinates[i][0]].y * frame.shape[0]))
                point_b = (int(landmarks[angle_coordinates[i][1]].x * frame.shape[1]),
                           int(landmarks[angle_coordinates[i][1]].y * frame.shape[0]))
                point_c = (int(landmarks[angle_coordinates[i][2]].x * frame.shape[1]),
                           int(landmarks[angle_coordinates[i][2]].y * frame.shape[0]))

                angle_obtained = calculate_angle(point_a, point_b, point_c, '0')

                # Compare with accurate angles
                if angle_obtained < accurate_angle_lists[i] - correction_value:
                    status = "more"
                elif angle_obtained > accurate_angle_lists[i] + correction_value:
                    status = "less"
                else:
                    status = "OK"
                    correct_angle_count += 1

                # Display status
                cv2.putText(frame, f"{angle_name_list[i]}: {status}", (point_b[0], point_b[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if status == "OK" else (0, 0, 255), 1)

            # Determine overall posture accuracy
            posture_status = "CORRECT" if correct_angle_count > 9 else "WRONG"
            posture_color = (0, 255, 0) if posture_status == "CORRECT" else (0, 0, 255)
            cv2.putText(frame, f"Posture: {posture_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert BGR back to RGB for Streamlit display
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames  # Return processed frames

