# Music Genre Classification


## Overview

This is the code for Music genre classification using Convolutional Neural Networks, implemented in Keras.

Comparison of classical approach of extracting features and use a classifier with modern approach of using CNNs on the audio representation (melspectrogram).

Deep Learning Approach:
1. Read the audio as melspectrogram, splitting them into 3s windows with 50% overlapping.
<br>Dataset size: 19000x129x128x1 (samples x time x frequency x channels).
2. Shuffle the input and split into train(70%) and test(30%).
3. Train the CNN and validate using validation set.


In case of VGG, channel requires 3 channels.


## Data

**Source:** http://opihi.cs.uvic.ca/sound/genres.tar.gz


## Dependencies

You can just run ``pip install -r requirements.txt`` in command prompt to install the necessary dependencies.

## Usage

``$ python gtzan.py -t test -m model/YOUR_MODEL_HERE -s data/YOUR_SONG_HERE``


## Results


|Model		    |Accuracy|
|-------------------|--------|
|Decision Tree      |0.508   |
|Logistic Regression|0.7     |
|Random Forest      |0.703   |
|SVM (RBF)   	    |0.762   |
|CNN 2D   	    |0.822   |
|VGG16   	    |0.857   |