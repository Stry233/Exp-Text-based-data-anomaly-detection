from scipy.io import wavfile
import os

import numpy as np
import torch
from librosa.feature import mfcc
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset

from datasets.preprocessing import get_target_label_idx


# Function to extract MFCCs
def extract_mfcc(audio_path):
    # Read the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # Extract MFCCs # TODO: align with the TIM-Net design(need check!!!)
    mfcc_features = mfcc(audio, sample_rate)

    return mfcc_features


class SAVEEDataset(Dataset):

    def __init__(self, root, normal_class='n', test_size=0.2, random_state=42):
        super().__init__()

        self.raw_data = np.load(root, allow_pickle=True).item()
        self.data = self.raw_data['x']
        self.labels = self.raw_data['y']

        # Assuming the labels are one-hot encoded, we convert them back to class indices
        self.labels = np.argmax(self.labels, axis=1)

        # Define the list of classes based on SAVEE dataset
        self.emotions = {
            "a": "anger",
            "d": "disgust",
            "f": "fear",
            "h": "happiness",
            "n": "neutral",
            "sa": "sadness",
            "su": "surprise"
        }

        self.normal_classes = tuple([k for k, v in self.emotions.items() if v == normal_class])
        self.outlier_classes = list(set(self.emotions.keys()) - set(self.normal_classes))

        # Do the scaling here
        # self.data = self._scale_features(self.data)

        # Split the dataset into training and test sets
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), int(self.labels[index] in self.outlier_classes), index

    def __len__(self):
        return len(self.data)

    def _scale_features(self, features):
        # Scale the features to have zero mean and unit variance
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) \
            -> (DataLoader, DataLoader):
        # Create data loaders for training and test sets
        train_indices = [i for i in range(len(self.train_data))]
        test_indices = [i for i in range(len(self.train_data), len(self.train_data) + len(self.test_data))]

        train_loader = DataLoader(Subset(self, train_indices), batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(Subset(self, test_indices), batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)

        return train_loader, test_loader