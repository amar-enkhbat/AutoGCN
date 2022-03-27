# Import utility libraries
import os
import numpy as np
from tqdm import tqdm
import random
import pickle
import json

# Import EEG preprocessing libraries 
# Import MNE library for loading/preprocessing Physionet EEG dataset 
from mne.channels import make_standard_montage
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne import events_from_annotations, Epochs, pick_types
from mne.decoding import Scaler
from sklearn.model_selection import train_test_split

# Import parameters
from params import PARAMS

# Define the random seed 
random_seed = PARAMS['RANDOM_SEEDS'][0]
random.seed(random_seed)
np.random.seed(random_seed)


def generate_preprocessed_data(subject_idc):
    
    """
    This function preprocesses all required trials then concatenates them into a single object

    The PhysioNet EEG MMI dataset has total of 14 runs and each run is saved as a separate .edf file (S**R01.edf, S**R14.edf etc). Each run has labels T0, T1, T2 however their meanings are defined as follows:
        Runs 1, 2:
            Baseline: eyes open, close
        Runs 3, 7, 11:
            Movement: T1: left hand vs T2: right hand
        Runs 5, 9, 13:
            Movement: T1: both hands vs T2: both feet

        Runs 4, 8, 12: 
            Motor imagery: T1: left vs T2: right hand
        Runs 6, 10, 14: 
            Motor imagery: T1: hands vs T2: feet
    For more details visit PhysioNet EEG MMI dataset homepage: https://physionet.org/content/eegmmidb/1.0.0/

    This work is intended for motor movement imagery classification therefore runs 1, 2, 3, 5, 7, 9, 11 and 13 are omitted. And imagery tasks are labeled as follows:
        label map:
        0: left hand
        1: right hand
        2: fist both hands
        3: feet

    Parameters
    ----------
    subject_idc : list
        List of subject indices to preprocess

    Returns
    -------
    (np.ndarray, np.ndarray)
        Returns tuple of numpy arrays: (data, label) for each subject with all imaginary runs
    """

    dataset_path = 'dataset/physionet.org/files/eegmmidb/1.0.0/'

    # Runs to extract (extract only imagery runs)
    run_idc = [4, 6, 8, 10, 12, 14]

    # Time delta
    delta = 1. / PARAMS['SAMPLING_RATE']

    # Start preprocessing and concatenating runs
    whole_data = []
    whole_label = []
    for subject_idx in subject_idc:
        subject_data = []
        subject_label = []
        for run_idx in run_idc:
            # Load files
            fname = os.path.join(dataset_path, f"S{subject_idx:03d}", f"S{subject_idx:03d}R{run_idx:02d}.edf")
            raw = read_raw_edf(fname, preload=True, verbose=False)

            eegbci.standardize(raw)  # set channel names
            montage = make_standard_montage('standard_1005') # Read a generic montage.
            raw.set_montage(montage) # Set channel positions and digitization points.
            raw.rename_channels(lambda x: x.strip('.')) # strip channel names of "." characters

            # Extract left hand vs right hand
            if run_idx in [4, 8, 12]:
                events, event_id = events_from_annotations(raw, event_id=dict(T1=0, T2=1), verbose=False)
            # Extract both hands vs both feet
            elif run_idx in [6, 10, 14]:
                events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)

            # Convert continuous data [num_samples, num_channels] into time-series slices [num_samples, num_channels, num_timesteps]. (For example: 1500x64 -> 15x64x100)
            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads') # Pick channels by type and names.
            epochs = Epochs(raw, events, event_id=event_id, tmin=0, tmax=PARAMS['SEQ_LEN'] / PARAMS['SAMPLING_RATE'] - delta, baseline=None, preload=True, proj=False, picks=picks, verbose=False) # Epochs extracted from a Raw instance.
            data = epochs.get_data() # Extract signal data
            label = epochs.events[:, -1] # Extract labels

            # Concatenate all run data into one data
            subject_data.append(data)
            subject_label.append(label)

        # Concatenate all run data into one data
        subject_data = np.vstack(subject_data)
        subject_label = np.hstack(subject_label)
        
        # Concatenate all subject data into one data 
        whole_data.append(subject_data)
        whole_label.append(subject_label)

    # Concatenate all subject data into one data 
    whole_data = np.vstack(whole_data).astype(np.float32)
    whole_label = np.hstack(whole_label)

    # Normalize whole data
    scaler = Scaler(scalings='mean')
    whole_data = scaler.fit_transform(whole_data)
    
    return whole_data, whole_label


def split_samples_105_subjects(n_splits):
    """
    Split 105 subject data into train/test datasets for total of n_splits=10 times then saved them as pickle files. Each split uses different random seed therefore all splits are different.

    ...
    Parameters
    ----------
    n_splits : int
        Number of different train/test datasets to create.
    """

    subject_idc = [i for i in range(1, 110)]
    subject_idc = [i for i in subject_idc if i not in PARAMS['EXCLUSIONS']]
    print('Number of subjects to use:', len(subject_idc))

    print('Loading data...')
    X, y = generate_preprocessed_data(subject_idc)
    print('Load complete.')

    print('Creating train/test datasets...')
    for i in tqdm(range(n_splits)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=PARAMS['RANDOM_SEEDS'][i])
        
        cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_105_subjects.pickle", "wb"))

        idc_dict = {'train_idc': subject_idc, 'test_idc': subject_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_105_subjects.json", "w"))
    print('DONE.')


def split_samples_50_subjects(n_splits):
    """
    Picks 50 random subject data the splits them into train/test datasets for total of n_splits times then saves them as pickle files. Each split uses different random seed therefore all splits are different.

    ...
    Parameters
    ----------
    n_splits : int
        Number of different train/test datasets to create.
    """

    subject_idc = [i for i in range(1, 110)]
    subject_idc = [i for i in subject_idc if i not in PARAMS['EXCLUSIONS']]
    subject_idc = random.sample(subject_idc, 50)
    print('Number of subjects to use:', len(subject_idc))

    print('Loading data...')
    X, y = generate_preprocessed_data(subject_idc)
    print('Load complete.')

    print('Creating train/test datasets...')
    for i in tqdm(range(n_splits)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=PARAMS['RANDOM_SEEDS'][i])
        
        cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_50_subjects.pickle", "wb"))

        idc_dict = {'train_idc': subject_idc, 'test_idc': subject_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_50_subjects.json", "w"))
    print('DONE.')


def split_samples_20_subjects(n_splits):
    """
    Picks 20 random subject data the splits them into train/test datasets for total of n_splits times then saves them as pickle files. Each split uses different random seed therefore all splits are different.

    ...
    Parameters
    ----------
    n_splits : int
        Number of different train/test datasets to create.
    """

    subject_idc = [i for i in range(1, 110)]
    subject_idc = [i for i in subject_idc if i not in PARAMS['EXCLUSIONS']]
    subject_idc = random.sample(subject_idc, 20)

    print('Number of subjects to use:', len(subject_idc))

    print('Loading data...')
    X, y = generate_preprocessed_data(subject_idc)
    print('Load complete.')

    print('Creating train/test datasets...')
    for i in tqdm(range(n_splits)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=PARAMS['RANDOM_SEEDS'][i])
        
        cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_20_subjects.pickle", "wb"))

        idc_dict = {'train_idc': subject_idc, 'test_idc': subject_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_20_subjects.json", "w"))
    print('DONE.')


def main():
    os.makedirs('./dataset/train', exist_ok=True)
    n_splits = PARAMS['N_RUNS']
    split_samples_105_subjects(n_splits)
    split_samples_50_subjects(n_splits)
    split_samples_20_subjects(n_splits)


if __name__ == '__main__':
    main()