
import os
import pathlib
import collections
import random
import shutil
import numpy as np
import pandas as pd
import random
import cv2


def list_files_per_class(folder_path):
    #retrieve all files from the local system

    videos = []

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.mp4'):
                videos.append(os.path.join(root, file_name))
    
    return videos

def get_class(fname):
    #retrieve the class name from the file name
    return fname.split('\\')[-2]

def get_files_per_class(files):
    #split the files into a dictionary where each class is the key and the vlues are a list of the corresponding files

    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def split_the_data(file_path, to_dir, file_names):
    #create a new directory and move the selected files there
    
    for file in file_names:
        class_name = get_class(file)
        src = os.path.join(file_path, class_name, file)

        des = pathlib.Path(to_dir) / class_name
        des.mkdir(parents=True, exist_ok=True)

        shutil.copy(src, des)

def split_class_lists(files_for_class, split_count):
    #splitting the files according to their category like training and testing while returning the remaining files for the other splits

    split_files = []
    remainder = {}
    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:split_count])
        remainder[cls] = files_for_class[cls][split_count:]
    return split_files, remainder

def download_ufc_crime(file_path, splits, download_dir):

    files = list_files_per_class(file_path)
    for f in files:
        tokens = f.split('\\')
        if len(tokens) <= 2:
            files.remove(f) # Remove that item from the list if it does not have a filename
  
    files_for_class = get_files_per_class(files)

    dirs = {}
    for split_name, split_count in splits.items():
        #get the videos according to each split and count and create a dictionary with the split name and directory path
        print(split_name, ":")
        split_dir = download_dir / split_name
        print(split_dir)
        split_files, files_for_class = split_class_lists(files_for_class, split_count)
        split_the_data(file_path, split_dir, split_files)
        dirs[split_name] = split_dir
    print(dirs)

    return dirs


def extract_frames(video_path):
    #Extracts frames from a video at equal intervals.

    print(f"Processing video: {video_path}")
    
    # Read the video
    video_capture = cv2.VideoCapture(str(video_path))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"No frames found in {video_path}. Skipping.")
        return []

    interval = 10

    # Initialize frame index and list to store extracted frames
    frame_index = 0
    key_frames = []

    # Loop over the video frames
    while frame_index < total_frames:
        # Set the frame index
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        success, curr_frame = video_capture.read()
        if not success:
            raise ValueError('Error reading the frame from the video')

        #resize frames
        curr_frame = cv2.resize(curr_frame, (64, 64))
        #save frames to seperate folder
        filename=os.path.join('U:\\Anomaly-Detection-Dataset\\Frames-Tags', f"{video_path.name}_{frame_index}.png")
        cv2.imwrite(filename, curr_frame)
        #add frame to key frame list
        key_frames.append(filename)

        # Increment frame index by the interval
        frame_index += interval

    # Release the video capture object
    video_capture.release()

    print(f"Total number of frames extracted from {video_path}: {len(key_frames)}")
    return key_frames



def FrameGenerator(path, training=False):

    def get_files_and_class_names():
        video_paths = list(path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    video_paths, classes = get_files_and_class_names()
    pairs = list(zip(video_paths, classes))

    if training:
        random.shuffle(pairs)

    for video_path, name in pairs:
        video_frames = extract_frames(video_path) 
        for frame in video_frames:
            yield frame, name


URL = 'U:\\Anomaly-Detection-Dataset\\Anomaly-Videos'
download_dir = pathlib.Path('U:\\Anomaly-Detection-Dataset\\Split-Data')
subset_paths = download_ufc_crime(URL, splits = {"train": 40, "test": 10}, download_dir = download_dir)


train_ds = pd.DataFrame(FrameGenerator(subset_paths['train'], training = True),columns=["Frames","Label"])

test_ds = pd.DataFrame(FrameGenerator(subset_paths['test']),columns=["Frames","Label"])