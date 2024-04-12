import pandas as pd

import os

df = pd.read_csv("extracted.csv")
df_copy = df.copy()

# Remplacer les chemins des fichiers dans la colonne 'file_path'
df_copy['filepath'] = df['filepath'].str.replace(
    r'UrbanSound8K/audio/fold(\d+)/', r'UrbanSound8K/audio/fold\1_aug/', regex=True)

df_final = pd.concat([df, df_copy])
print(len(df_final))
print(df_final.head())
print(df_final.tail())

AUDIO_FOLDER = "UrbanSound8K/audio"
audio_folder_content = os.listdir(AUDIO_FOLDER)

# # Remove '.DS_Store' from the list
# if '.DS_Store' in audio_folder_content:
#     del audio_folder_content[audio_folder_content.index('.DS_Store')]
# fold_paths = [os.path.join(AUDIO_FOLDER, rep) for rep in audio_folder_content]
# for fold_path in fold_paths:
#     if fold_path.endswith("_aug"):
#         for audio_file_path in os.listdir(fold_path):
#             if audio_file_path.endswith('.wav_aug'):
#                 complete_audio_file_path = os.path.join(
#                     fold_path, audio_file_path)
#                 filename_without_extension = os.path.splitext(
#                     complete_audio_file_path)[0]
#                 os.rename(complete_audio_file_path,
#                           filename_without_extension + '_aug.wav')
