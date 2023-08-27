import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import os
import subprocess

# Load the trained model
loaded_model = load_model("path_to_your_model.h5")

IMAGE_HEIGHT, IMAGE_WIDTH = 100, 100

# List of sports action class labels
CLASSES_LIST = [
    "Basketball", "Biking", "GolfSwing", "IceDancing", "JugglingBalls",
    "HorseRiding", "LongJump", "Shotput", "TennisSwing", "TaiChi"
]

SEQUENCE_LENGTH = 20


def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    # Open the video file for reading
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Open the video file for writing with the specified codec and FPS
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    
    # Queue to hold frames for sequence processing
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read()

        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            # Predict the sports action class based on the sequence of frames
            predicted_labels_probabilities = loaded_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(frame)

    # Release video resources
    video_reader.release()
    video_writer.release()


def main():
    st.title('GameSync Web App')
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mpeg"])

    if uploaded_file is not None:
        output_file_path = os.path.join("output_videos/",
                                        uploaded_file.name.split("/")[-1].split(".")[0] + "_output.mp4")
        with st.spinner('Processing video...'):
            predict_on_video("input_videos/" + uploaded_file.name.split("/")[-1], output_file_path, SEQUENCE_LENGTH)

            os.makedirs("output_videos", exist_ok=True)
            os.chdir('output_videos')
            subprocess.call(['ffmpeg', '-y', '-i', uploaded_file.name.split("/")[-1].split(".")[0] + ".mp4",
                             '-vcodec', 'libx264', '-f', 'mp4', 'output.mp4'], shell=True)
            st.success('Processing complete!')

            video_path = os.path.join("output_videos/" + uploaded_file.name.split("/")[-1].split(".")[0] + "_output.mp4")
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

    else:
        st.text("Please upload a video file")


if __name__ == '__main__':
    main()
