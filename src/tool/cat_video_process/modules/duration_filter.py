import cv2
import os


def check_video_duration(video_path, min_duration):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        return False

    duration = frame_count / fps

    return duration >= min_duration