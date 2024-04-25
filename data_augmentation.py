"""
This script is designed to augment the volume of data using the original UrbanSound8k 
dataset located in the directory with the same name by altering the original sounds.

The purpose of this augmentation is to increase the diversity of the dataset by shifting 
original sounds for improving the generalization of machine learning models trained on it.

Usage:
    python data_augmentation.py

Note: Before running this script, ensure that the UrbanSound8k dataset is located 
in the same directory as this script.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from audiomentations import Compose, Shift
import shutil

augment = Compose([
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

AUDIO_FOLDER = "UrbanSound8K/audio"
audio_folder_content = os.listdir(AUDIO_FOLDER)

# Remove '.DS_Store' from the list
if '.DS_Store' in audio_folder_content:
    del audio_folder_content[audio_folder_content.index('.DS_Store')]
fold_paths = [os.path.join(AUDIO_FOLDER, rep)
              for rep in audio_folder_content]
for fold_path in fold_paths:
    for audio_file_path in os.listdir(fold_path):
        # # Load
        if audio_file_path.endswith('.wav'):
            complete_audio_file_path = os.path.join(fold_path, audio_file_path)
            print(complete_audio_file_path)
            audio_data, sample_rate = librosa.load(
                complete_audio_file_path, sr=None, mono=True)
            audio_data = audio_data.astype(np.float32)
            # Add effects
            augmented_audio = augment(
                samples=audio_data, sample_rate=sample_rate)
            # Save
            filename_without_extension = os.path.splitext(audio_file_path)[0]
            output_file_name = filename_without_extension + "_aug.wav"
            output_folder = fold_path + "_aug"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            complete_save_path = os.path.join(output_folder, output_file_name)
            sf.write(complete_save_path, augmented_audio, sample_rate)
