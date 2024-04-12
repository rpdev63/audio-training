from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import librosa
import soundfile as sf
import os

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])


AUDIO_FOLDER = "UrbanSound8K/audio"
audio_folder_content = os.listdir(AUDIO_FOLDER)

# Remove '.DS_Store' from the list
if '.DS_Store' in audio_folder_content:
    del audio_folder_content[audio_folder_content.index('.DS_Store')]
fold_paths = [os.path.join(AUDIO_FOLDER, rep) for rep in audio_folder_content]
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

            # # Load the WAV file
            # file_path = "UrbanSound8K/audio/fold1/7061-6-0-0.wav"
            # audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)

        # # Ensure the audio data is in the correct format (floating-point samples with a sample rate of 16000 Hz)
        # audio_data = audio_data.astype(np.float32)

        # # Augment the audio data
        # augmented_audio = augment(samples=audio_data, sample_rate=sample_rate)
        # # Save the augmented audio to a new WAV file
        # output_file_path = "output_audio_augmented.wav"
        # output_file_path2 = "output_audio_augmented2.wav"

        # augmented_audio2 = augment2(samples=audio_data, sample_rate=sample_rate)

        # sf.write(output_file_path, augmented_audio, sample_rate)
        # sf.write(output_file_path2, augmented_audio2, sample_rate)
