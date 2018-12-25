# Music Genre Classification


## Overview

This is the code for Music genre classification using Convolutional Neural Networks, implemented in Keras.


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