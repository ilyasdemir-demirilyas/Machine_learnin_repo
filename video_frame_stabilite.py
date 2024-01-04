import os
import cv2
from moviepy.editor import VideoFileClip, VideoClip
import moviepy.video.fx.all as vfx

def process_videos(target_length, input_folder, output_folder):
    # Döngü ile her bir videoyu işleyin
    for video_file in os.listdir(input_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(input_folder, video_file)

            # Open the video file
            video = cv2.VideoCapture(video_path)

            # Get the length of the video in frames
            current_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # Get the current frames per second (fps) of the video
            current_fps = video.get(cv2.CAP_PROP_FPS)

            # Calculate the new fps
            new_fps = current_length / target_length

            # Release the video capture object
            video.release()

            # Import video clip
            clip = VideoFileClip(video_path)

            # Apply speed up
            final = clip.fx(vfx.speedx, new_fps)

            # Save video clip
            out_loc = os.path.join(output_folder, f"{video_file.split('.')[0]}_output.mp4")
            final.write_videofile(out_loc)

    print("Processing completed for all videos.")

# Kullanım örneği
target_length = 60
input_folder = r'E:\Lstm_son_deneme_siyah_beyaz\abla'
output_folder = r'E:\Lstm_son_deneme_siyah_beyaz\abla'

process_videos(target_length, input_folder, output_folder)
