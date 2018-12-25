import numpy as np
import os
import gc
import argparse
import librosa
import keras
from keras import backend as K
from keras.models import load_model

# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

def splitsongs(X, window = 0.1, overlap = 0.5):
    temp_X = []
    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
    return np.array(temp_X)

def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]
    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

# Helper function to get key by value from a dictionary
def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys

def main(args):
    # Check if the file path to the model was passed
    if not args.model:
        raise ValueError("File path to model should be passed in test mode.")
    # Check if was passed the music file
    if not args.song:
        raise ValueError("Song path should be passed in test mode.")
    model = load_model(args.model)
    X, sr = librosa.load(args.song)
    X = X[:song_samples]
    X = splitsongs(X)
    X = to_melspectrogram(X)
    pred = model.predict(X)
    pred = np.argmax(pred, axis=1)
    pred = np.bincount(pred).argmax()
    print(getKeysByValue(genres, pred))

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Classification')
    # Required arguments
    parser.add_argument('-t', '--type', help='train or test mode to execute', type=str, required=True)
    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('-d', '--directory', help='Path to the root directory with GTZAN files', type=str)
    parser.add_argument('-m', '--model', help='If test is chosen, path to trained model', type=str)
    parser.add_argument('-s', '--song', help='If test is chosen, path to song to classify', type=str)
    args = parser.parse_args()
    # Call the main function
    main(args)