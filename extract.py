import pandas as pd
import os
import librosa
import numpy as np
import time
from tqdm import tqdm
import argparse


class FeatureExtractor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.max_audio_duration = 4
        self.dataset_df = self._create_dataset(csv_file)

    @staticmethod
    def _create_dataset(csv_file):
        """
        Args:
            dataset_path: path with the .wav files after unzipping
        Returns: A pandas dataframe with the list of files and labels (`filenames`, `labels`)
        """
        dataset_df = pd.read_csv(csv_file)
        filepaths = []
        for i, row in dataset_df.iterrows():
            filepaths.append(os.path.join('UrbanSound8K/audio',
                             'fold'+str(row['fold']), row['slice_file_name']))
        dataset_df['filepath'] = filepaths
        return dataset_df

    @staticmethod
    def _compute_max_pad_length(max_audio_length, sample_rate=22050, n_fft=2048, hop_length=512):
        dummy_file = np.random.random(max_audio_length*sample_rate)
        stft = librosa.stft(dummy_file, n_fft=n_fft, hop_length=hop_length)
        # Return an even number for CNN computation purposes
        if stft.shape[1] % 2 != 0:
            return stft.shape[1] + 1
        return stft.shape[1]

    def compute_save_features(self,
                              mode='mfcc',
                              sample_rate=22050,
                              n_fft=2048,
                              hop_length=512,
                              n_mfcc=40,
                              deltas=False
                              ):
        output_path = f'features_{mode}'
        max_pad = self._compute_max_pad_length(self.max_audio_duration,
                                               sample_rate=sample_rate,
                                               n_fft=n_fft,
                                               hop_length=hop_length)
        print('Max Padding = ', max_pad)

        if not os.path.exists(output_path):
            print('Creating output folder: ', output_path)
            os.makedirs(output_path)
        else:
            print('Output folder already existed')

        print('Saving features in ', output_path)
        start = time.perf_counter_ns()
        features_path = []
        for filepath in tqdm(self.dataset_df['filepath']):
            # if i % 100 == 0:
            #     print('{} files processed in {}s'.format(i, time.time() - t))
            audio_file, sample_rate = librosa.load(filepath, sr=sample_rate)
            if mode == 'mfcc':
                audio_features = self.compute_mfcc(
                    audio_file, sample_rate, n_fft, hop_length, n_mfcc, deltas)
            elif mode == 'stft':
                audio_features = self.compute_stft(
                    audio_file, sample_rate, n_fft, hop_length)
            elif mode == 'mel':
                audio_features = self.compute_mel_spectogram(
                    audio_file, sample_rate, n_fft, hop_length)

            audio_features = np.pad(audio_features,
                                    pad_width=((0, 0), (0, max_pad - audio_features.shape[1])))

            save_path = os.path.join(
                output_path, filepath.split('/')[-1].replace('wav', 'npy'))
            self.save_features(audio_features, save_path)
            features_path.append(save_path)
        end = time.perf_counter_ns()
        print(f'Temps d\'éxécution : {(end - start) / 1e9} s\n')
        self.dataset_df['features_path'] = features_path
        return self.dataset_df

    @staticmethod
    def save_features(audio_features, filepath):
        np.save(filepath, audio_features)

    @staticmethod
    def compute_mel_spectogram(audio_file, sample_rate, n_fft, hop_length):
        return librosa.feature.melspectrogram(y=audio_file,
                                              sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length)

    @staticmethod
    def compute_stft(audio_file, sample_rate, n_fft, hop_length):
        return librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length)

    @staticmethod
    def compute_mfcc(audio_file, sample_rate, n_fft, hop_length, n_mfcc, deltas=False):
        mfccs = librosa.feature.mfcc(y=audio_file,
                                     sr=sample_rate,
                                     n_fft=n_fft,
                                     n_mfcc=n_mfcc,
                                     )
        # Change mode from interpolation to nearest
        if deltas:
            delta_mfccs = librosa.feature.delta(mfccs, mode='nearest')
            delta2_mfccs = librosa.feature.delta(
                mfccs, order=2, mode='nearest')
            return np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        return mfccs


if __name__ == '__main__':
    fe = FeatureExtractor('UrbanSound8K/metadata/UrbanSound8K.csv')
    file_path = "extracted.csv"
    features_name = "mfcc"

    dataset_df = fe.compute_save_features(
        mode=features_name,
        n_mfcc=13,
        deltas=True
    )

    if os.path.exists(file_path):
        extracted_df = pd.read_csv(file_path)
        extracted_df[f"{features_name}_features_path"] = dataset_df["features_path"]
    else:
        extracted_df = dataset_df.rename(
            columns={"features_path": f"{features_name}_features_path"})
    extracted_df.to_csv(file_path, index=False)
