import cv2
import os
import numpy as np

def extract_frames_from_videos(video_folder, output_folder, label, num_frames=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(video_folder):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        count = 0
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # jump to the frame
            ret, frame = cap.read()
            if not ret:
                continue
            frame_name = f"{label}_{video_file[:-4]}_frame{count}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            count += 1

        cap.release()
        print(f"Extracted {count} frames from {video_file}")

extract_frames_from_videos("frames/fake", "extracted_frames/fake", "fake", num_frames=20)
extract_frames_from_videos("frames/real", "extracted_frames/real", "real", num_frames=20)
