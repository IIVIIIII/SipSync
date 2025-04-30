import os
import numpy as np
import random
import tensorflow as tf
import keras
from tcn import TCN
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import librosa
import librosa.display
import mirdata


eps = np.finfo(float).eps

# mood_mapping = {'mood/theme---happy': 0,
#  'mood/theme---positive': 1,
#  'mood/theme---love': 2,
#  'mood/theme---romantic': 3,
#  'mood/theme---energetic': 4,
#  'mood/theme---party': 5,
#  'mood/theme---relaxing': 6,
#  'mood/theme---calm': 7,
#  'mood/theme---sad': 8,
#  'mood/theme---melancholic': 9,
#  'mood/theme---dark': 10,
#  'mood/theme---heavy': 11,
#  'mood/theme---dream': 12,
#  'mood/theme---space': 13,
#  'mood/theme---inspiring': 14,
#  'mood/theme---hopeful': 15}

mood_mapping = {'mood/theme---happy': 0,
 'mood/theme---positive': 1,
 'mood/theme---love': 2,
 'mood/theme---romantic': 3,
 'mood/theme---energetic': 4,
 'mood/theme---party': 5,
 'mood/theme---relaxing': 6,
 'mood/theme---calm': 7,
 'mood/theme---sad': 8,
 'mood/theme---melancholic': 9}


# first8
# {'mood/theme---happy': 0,
#  'mood/theme---positive': 1,
#  'mood/theme---love': 2,
#  'mood/theme---romantic': 3,
#  'mood/theme---energetic': 4,
#  'mood/theme---party': 5,
#  'mood/theme---relaxing': 6,
#  'mood/theme---calm': 7}

#better 8
# mood_mapping = {'mood/theme---happy': 0,
#  'mood/theme---love': 1,
#  'mood/theme---energetic': 2,
#  'mood/theme---relaxing': 3,
#  'mood/theme---sad': 4,
#  'mood/theme---dark': 5,
#  'mood/theme---dream': 6,
#  'mood/theme---inspiring': 7}


def load_data(data_home, dataset_name='mtg_jamendo_autotagging_moodtheme', version='default', track_ids=None):
    """
    Load data from a specified music dataset and return the audio file paths
    and their corresponding labels.

    Parameters
    ----------
    data_home : str
        The root directory where the dataset is stored.
    dataset_name : str, optional
        The name of the dataset to load, by default 'gtzan_genre'.
    version : str, optional
        The version of the dataset to load, by default '1.0'.
    track_ids : list of str, optional
        A list of track IDs to load from the dataset, by default None. If None, all tracks in the dataset will be loaded.

    Returns
    -------
    audio_file_paths : list of str
        A list of audio file paths from the specified dataset.
    labels : list of int
        A list of corresponding labels for the audio files.

    Example
    -------
    >>> data_home = "/path/to/data_directory"
    >>> dataset_name = "gtzan_genre"
    >>> version = "mini"
    >>> track_ids = ["track_1", "track_2"]
    >>> audio_file_paths, labels = load_data(data_home, dataset_name, version, track_ids)
    """

    dataset = mirdata.initialize(dataset_name,
                                data_home=data_home.decode('utf8'),
                                version=version)
    ids = dataset.track_ids
    audio_file_paths = []
    labels = []
    if track_ids is not None:
        ids = [cid.decode('utf8') for cid in track_ids]
    for fid in ids:
        track = dataset.track(fid)
        audio_file_paths.append(track.audio_path)
        labels.append(mood_mapping[track.tags])

    return audio_file_paths, labels


def compute_mel_spectrogram(audio, sample_rate=22050, n_mels=128, hop_length=512):
    """
    Compute the normalized Mel spectrogram of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal as a 1D numpy array.
    sample_rate : int, optional
        Sampling rate of the audio signal, by default 22050.
    n_mels : int, optional
        Number of Mel bands to generate, by default 128.
    hop_length : int, optional
        Number of samples between successive frames, by default 512.

    Returns
    -------
    np.ndarray
        Mel spectrogram as a 2D numpy array.

    """
    # Hint: use librosa melspectrogram and librosa power_to_db
    # YOUR CODE HERE

    # Compute Mel spectrogram from the audio signal
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    
    # Convert the Mel spectrogram to dB scale
    melspec = librosa.power_to_db(melspec)

    return melspec

    pass


def window_audio(audio, sample_rate, audio_seg_size, segments_overlap):
    """
    Segment audio into windows with a specified size and overlap.
    Padding is added only to the last window.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal to be segmented.
    sample_rate : int
        The sampling rate of the audio signal.
    audio_seg_size : float
        The duration of each window in seconds.
    segments_overlap : float
        The duration of the overlap between consecutive windows in seconds.

    Returns
    -------
    audio_windows : list of np.ndarray
        A list of windows of the audio signal.

    Example
    -------
    >>> import librosa
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> audio_windows = window_audio(y, sr, audio_seg_size=1, segments_overlap=0.5)
    """
    # YOUR CODE HERE

    windows = []

    # Calculate the window size in samples
    win_size = audio_seg_size * sample_rate
    
    # Calculate the overlap size in samples
    overlap_size = segments_overlap * sample_rate

    # Iterate through the audio signal, extracting windows


    num_windows = int(np.ceil(len(audio)/(win_size-overlap_size)))


    padding = np.zeros(len(audio) % (win_size*num_windows))
    padded = np.concatenate([audio, padding])

    
    for hop in range(num_windows):
      if hop+win_size < len(padded):
        hop = int(hop * (win_size-overlap_size))
        window = padded[hop: hop+win_size]
        windows.append(window)


        # If the window end is within the audio length, extract the window
        
            # Padding the last window with zeros if it extends beyond the audio length

        # Add the window to the list of audio windows
        
        # Update the start position for the next window, considering the overlap

    return windows

    # pass


def pitch_shift_audio(audio, sample_rate=22050, pitch_shift_steps=2):
    """
    Augments the audio signal by applying pitch shifting.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal to be augmented.
    sample_rate : int, optional, default: 22050
        The sample rate of the audio signal.
    pitch_shift_steps : int, optional, default: 2
        The number of half-steps to shift the pitch (positive or negative).

    Returns
    -------
    np.ndarray
        The pitch-shifted audio signal.
    """
    # Hint: use librosa pitch shift
    # YOUR CODE HERE

    shifty = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift_steps)

    return shifty

    pass


def extract_yamnet_embedding(wav_data, yamnet):
    """
    Run YAMNet to extract embeddings from the wav data.

    Parameters
    ----------
    wav_data : np.ndarray
        The audio signal to be processed.
    yamnet : tensorflow.keras.Model
        The pre-trained YAMNet model.

    Returns
    -------
    np.ndarray
        The extracted embeddings from YAMNet.
    """
    # Hint: check the tensorflow models to see how YAMNET should be used
    # YOUR CODE HERE

    scores, embeddings, spectrogram = yamnet(wav_data)

    return embeddings

    pass



# def extract_musicfm_embedding(wav_data, musicfm):
#     """
#     Run YAMNet to extract embeddings from the wav data.

#     Parameters
#     ----------
#     wav_data : np.ndarray
#         The audio signal to be processed.
#     yamnet : tensorflow.keras.Model
#         The pre-trained YAMNet model.

#     Returns
#     -------
#     np.ndarray
#         The extracted embeddings from YAMNet.
#     """

#     # wav_data = tf.cast(wav_data, tf.float32)

#     embeddings = musicfm.get_latent(wav_data, layer_ix=7)

#     print(type(embeddings))

#     return embeddings

#     pass





def wav_generator(data_home, augment, track_ids=None, sample_rate=22050,
                  pitch_shift_steps=2, shuffle=True):
    """
    Generator function that yields audio and labels from the specified dataset,
    with optional data augmentation.

    Parameters
    ----------
    data_home : str
        The root directory where the dataset is stored.
    augment : bool
        Whether to apply data augmentation (pitch shifting) to the audio.
    track_ids : list of str, optional
        A list of track IDs to load from the dataset, by default None.
        If None, all tracks in the dataset will be loaded.
    sample_rate : int, optional
        The sample rate at which to load the audio, by default 22050.
    pitch_shift_steps : int, optional
        The number of steps by which to shift the pitch for data augmentation, by default 2.
    shuffle : bool, optional
        Whether to shuffle the data before iterating, by default True.

    Yields
    ------
    audio : np.ndarray
        A NumPy array containing the audio waveform data.
    label : int
        The corresponding label for the audio.

    Example
    -------
    >>> data_home = "/path/to/data_directory"
    >>> augment = True
    >>> track_ids = ["track_1", "track_2"]
    >>> generator = wav_generator(data_home, augment, track_ids)
    >>> for audio, label in generator:
    ...     # Process audio and label

    """
    # Hint: base your generator on the win_generator
    # YOUR CODE HERE

    # Get list of audio paths and their corresponding labels
    audio_file_paths, labels = load_data(data_home, track_ids=track_ids)

    # Convert labels to numpy array
    labels = np.array(labels)

    # Shuffle data
    if shuffle:
        idxs = np.random.permutation(len(labels))
        audio_file_paths = [audio_file_paths[i] for i in idxs]
        labels = labels[idxs]


    for idx in range(len(audio_file_paths)):

        # Load audio at given sample_rate and label
        label = labels[idx]
        audio, _ = librosa.load(audio_file_paths[idx], sr=sample_rate)

        # Shorten audio to 29s due to imprecisions in duration of GTZAN
        # (ensures same duration files)
        audio = audio[:29*sample_rate]

        # Apply augmentation
        if augment:
            audio = pitch_shift_audio(audio, sample_rate, pitch_shift_steps)


        
        # audio = tf.cast(audio, tf.float32) #REMOVE


        yield audio, label

    pass


def win_generator(data_home, augment, track_ids=None, sample_rate=22050, pitch_shift_steps=2,
                   n_mels=128, hop_length=512, audio_seg_size=1, segments_overlap=0.5, shuffle=True):
    """
    Generator function that yields Mel spectrograms and labels from the specified dataset,
    with optional data augmentation. 
    Audio is broken down in small windows, the spectrogram is computed and yielded along with the label. 
    The label of the window is assumed to be the same as the label for the entire track.

    Parameters
    ----------
    data_home : str
        The root directory where the dataset is stored.
    augment : bool
        Whether to apply data augmentation (pitch shifting) to the audio.
    track_ids : list of str, optional
        A list of track IDs to load from the dataset, by default None.
        If None, all tracks in the dataset will be loaded.
    sample_rate : int, optional
        The sample rate at which to load the audio, by default 22050.
    pitch_shift_steps : int, optional
        The number of steps by which to shift the pitch for data augmentation, by default 2.
    n_mels : int, optional
        The number of Mel bands to generate, by default 128.
    hop_length : int, optional
        The number of samples between successive frames, by default 512.
    audio_seg_size : float, optional
        The size of audio segments in seconds, by default 1.
    segments_overlap : float, optional
        The overlap between audio segments in seconds, by default 0.5.
    shuffle : bool, optional
        Whether to shuffle the data before iterating, by default True.

    Yields
    ------
    spectrogram : np.ndarray
        A NumPy array containing the Mel spectrogram data.
    label : int
        The corresponding label for the spectrogram.

    Example
    -------
    >>> data_home = "/path/to/data_directory"
    >>> augment = True
    >>> track_ids = ["track_1", "track_2"]
    >>> generator = win_generator(data_home, augment, track_ids)
    >>> for spectrogram, label in generator:
    ...     # Process spectrogram and label
    """

    # Get list of audio paths and their corresponding labels
    audio_file_paths, labels = load_data(data_home, track_ids=track_ids)

    # Convert labels to numpy array
    labels = np.array(labels)

    # Shuffle data
    if shuffle:
        idxs = np.random.permutation(len(labels))
        audio_file_paths = [audio_file_paths[i] for i in idxs]
        labels = labels[idxs]


    for idx in range(len(audio_file_paths)):

        # Load audio at given sample_rate and label
        label = labels[idx]
        audio, _ = librosa.load(audio_file_paths[idx], sr=sample_rate)

        # Shorten audio to 29s due to imprecisions in duration of GTZAN
        # (ensures same duration files)
        audio = audio[:29*sample_rate]

        # Apply augmentation
        if augment:
            audio = pitch_shift_audio(audio, sample_rate, pitch_shift_steps)

        # Compute audio windowing
        audio_windows = window_audio(audio, sample_rate, audio_seg_size, segments_overlap)

        # Loop over windows
        for window in audio_windows:
            
            # Compute Mel spectrogram
            spectrogram = compute_mel_spectrogram(window, sample_rate, n_mels, hop_length)
            spectrogram = np.expand_dims(spectrogram, axis=2)

            yield spectrogram, label


def create_dataset(data_generator, input_args, input_shape):
    """
    Create a TensorFlow dataset from a data generator function along with the
    specified input arguments and shape.

    Parameters
    ----------
    data_generator : callable
        The data generator function to use for creating the dataset.
    input_args : list
        A list containing the arguments to be passed to the data generator function.
    input_shape : tuple
        A tuple representing the shape of the input data.

    Returns
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset created from the data generator function.

    Example
    -------
    >>> def sample_generator():
    ...     for i in range(10):
    ...         yield np.random.random((4, 4)), i
    >>> input_args = []
    >>> input_shape = (4, 4, 1)
    >>> dataset = create_dataset(sample_generator, input_args, input_shape)
    """

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=input_args,
        output_signature=(
          tf.TensorSpec(shape=input_shape, dtype=tf.float32),
          tf.TensorSpec(shape=(), dtype=tf.uint8)
          )
        )

    return dataset


def reload_tcn(model_path, weights_path, optimizer, loss, metrics):
    """
    Reload a TCN model from a JSON file and restore its weights. 
    Preferred method when dealing with custom layers.

    Parameters
    ----------
    model_path : str
        The path to the JSON file containing the model architecture.
    weights_path : str
        The path to the model weights file.
    optimizer : str or tf.keras.optimizers.Optimizer
        The optimizer to use when compiling the model.
    loss : str or tf.keras.losses.Loss
        The loss function to use when compiling the model.
    metrics : list of str or tf.keras.metrics.Metric
        The list of metrics to use when compiling the model.

    Returns
    -------
    reloaded_model : tf.keras.Model
        The reloaded model with the restored weights.

    Example
    -------
    >>> model_path = 'path/to/saved_model.json'
    >>> weights_path = 'path/to/saved_weights.h5'
    >>> optimizer = 'adam'
    >>> loss = 'sparse_categorical_crossentropy'
    >>> metrics = ['accuracy']
    >>> reloaded_model = reload_tcn(model_path, weights_path, optimizer, loss, metrics)
    """
    # Load the best checkpoint of the model from json file (due to custom layers)
    loaded_json = open(model_path, 'r').read()
    reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

    reloaded_model.compile(optimizer=optimizer,
                            loss=loss, 
                            metrics=metrics)
    # restore weights
    reloaded_model.load_weights(weights_path)

    return reloaded_model


# ============= Auxiliary Functions: DO NOT CHANGE =============


def explore_data_mood(labels_train, labels_val, labels_test, mood_mapping):
    """
    Plot the class distributions for the training, validation, and test sets based on mood tags.
    """
    class_names = list(mood_mapping.keys())
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def map_labels(labels):
        return [mood_mapping[tag] for tag in labels if tag in mood_mapping]

    # Train
    ax[0].bar(range(len(class_names)), np.bincount(map_labels(labels_train), minlength=len(class_names)))
    ax[0].set_title("Training Set")
    ax[0].set_xticks(range(len(class_names)))
    ax[0].set_xticklabels(class_names, rotation=90)

    # Validation
    ax[1].bar(range(len(class_names)), np.bincount(map_labels(labels_val), minlength=len(class_names)))
    ax[1].set_title("Validation Set")
    ax[1].set_xticks(range(len(class_names)))
    ax[1].set_xticklabels(class_names, rotation=90)

    # Test
    ax[2].bar(range(len(class_names)), np.bincount(map_labels(labels_test), minlength=len(class_names)))
    ax[2].set_title("Test Set")
    ax[2].set_xticks(range(len(class_names)))
    ax[2].set_xticklabels(class_names, rotation=90)

    plt.tight_layout()
    plt.show()

def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def plot_spectrogram(log_spectrogram, sr, hop_length, genre=None):
    """
    Plot the Mel spectrogram of an audio signal.

    Parameters
    ----------
    log_spectrogram : np.ndarray
        The log-scaled Mel spectrogram of the audio signal (output of `librosa.power_to_db()`).
    sr : int
        The sampling rate of the audio signal.
    hop_length : int
        The number of samples between successive frames in the spectrogram computation.
    genre : str, optional
        The genre of the audio signal (default is None). If provided, it will be added to the plot title.

    Returns
    -------
    None

    Example
    -------
    >>> sr = 22050
    >>> hop_length = 512
    >>> log_S = compute_mel_spectrogram(audio, sample_rate=sr, n_mels=128, hop_length=hop_length)
    >>> plot_spectrogram(log_S, sr, hop_length=hop_length, genre='Jazz')
    """
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                              x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram {genre}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

