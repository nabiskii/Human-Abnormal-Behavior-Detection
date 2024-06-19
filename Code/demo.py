import tkinter as tk
import tensorflow as tf
from tkinter import filedialog, scrolledtext
import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from pathlib import Path
import statistics as stats

np.set_printoptions(threshold=np.inf)
global key

IMG_HEIGHT = 64
IMG_WIDTH = 64
SEED = 12
BATCH_SIZE = 64

preprocess_fun = preprocess_input

test_dir = Path("D:/Vscode/Python/final year project/Test_video_Frames/Testing")

# Load the model without loading the optimizer
model = load_model('D:/Vscode/Python/final year project/model/model_new1.keras', compile=False)

# Recreate the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.00003)

# Compile the model with the new optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def open_video_dialog():
    video_path = Path(filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]))
    if video_path:
        process_video(video_path)

def process_video(video_path):
    print(f"Processing video: {video_path}")
    Key_frames = video_path
    video_capture = cv2.VideoCapture(str(video_path))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"No frames found in {video_path}. Skipping.")
        return []

    frame_index = 0
    key_frames = []
    while frame_index < total_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, curr_frame = video_capture.read()
        if not success:
            raise ValueError('Error reading the frame from the video')
        
        curr_frame_resized = cv2.resize(curr_frame, (IMG_WIDTH, IMG_HEIGHT))
        
        # Save frame as PNG
        filename = f"{test_dir}/{video_path.stem}_{frame_index}.png"
        key_frames.append(curr_frame_resized)
        cv2.imwrite(filename, curr_frame_resized)
        
        frame_index += 10

    video_capture.release()
    process_prediction(Key_frames)

def get_class_name(class_):
    if class_ == 0:
        return "Assault"
    elif class_ == 1:
        return "Explosion"
    elif class_ == 2:
        return "Fighting"
    elif class_ == 3:
        return "Normal"
    elif class_ == 4:
        return "Road Accident"
    elif class_ == 5:
        return "Robbery"
    else:
        return "Vandalism"

def process_prediction(key):

    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_fun)
    folder = "D:/Vscode/Python/final year project/Test_video_Frames"
    test_generator = test_datagen.flow_from_directory(directory=folder,
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False,
                                                      color_mode="rgb",
                                                      class_mode=None,
                                                      seed=SEED)
    

    test_generator.reset()
    predictions = model.predict(test_generator)

    y_preds = np.argsort(predictions, axis=1)[:,6]

    # Get final prediction
    final_pred = get_class_name(stats.mode(y_preds))
    
    output_text.config(state='normal')
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, f"Prediction: {final_pred}\n")
    output_text.config(state='disabled')

root = tk.Tk()
root.title("Video Input Dialog")

open_dialog_button = tk.Button(root, text="Open Video Dialog", command=open_video_dialog)
open_dialog_button.pack(pady=10)
output_text = scrolledtext.ScrolledText(root, width=40, height=10)
output_text.pack(pady=10)
output_text.config(state='disabled')

root.mainloop()
