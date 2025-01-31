import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from utils import evaluate, correct_feedback

# Streamlit UI
st.title("Yoga Pose Detection & Correction")

# File uploader for training and test data
train_file = st.file_uploader("Upload Training CSV", type=["csv"])
test_file = st.file_uploader("Upload Test CSV", type=["csv"])

if train_file and test_file:
    # Load data
    data_train = pd.read_csv(train_file)
    data_test = pd.read_csv(test_file)

    # Prepare features and labels
    X, Y = data_train.iloc[:, :-1], data_train['target']

    # Train the SVM model
    st.write("Training the SVM model...")
    model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
    model.fit(X, Y)
    st.success("Model trained successfully!")

    # Evaluate the model
    st.write("Evaluating the model...")
    predictions = evaluate(data_test, model, show=False)

    # Create a confusion matrix
    cm = confusion_matrix(data_test['target'], predictions)

    # Display the confusion matrix using a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# File uploader for video input
video_file = st.file_uploader("Upload Video for Pose Correction", type=["mp4"])
teacher_csv = st.file_uploader("Upload Teacher Angles CSV", type=["csv"])

if video_file and teacher_csv:
    # Save uploaded files
    video_path = f"./{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    teacher_csv_path = f"./{teacher_csv.name}"
    with open(teacher_csv_path, "wb") as f:
        f.write(teacher_csv.read())

    # Run pose correction
    st.write("Processing video for pose correction...")
    correct_feedback(model, video_path, teacher_csv_path)
    st.success("Pose correction completed!")

cv2.destroyAllWindows()
