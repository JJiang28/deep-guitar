'''
This file holds functions relating to file formatting
'''

import os
import sys
from moviepy.editor import VideoFileClip

def convert_mp4_to_wav(video_file, output_path):

    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(output_path)
    
