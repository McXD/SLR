# Sign Language Recognition

## Introduction

A sign language recognition program for 24 letters (excluding J and Z). The program has five modes of operation, based on the command-line parameter:

1. Collect Images
2. Extract Features
3. Visualize Features
4. Train Classifier
5. Play Game

The flow to use the program should be like this:

- First, collect images where the user should pose the gesture and type the corresponding character on the keyboard. The images are stored in a sub-directory with the corresponding letter as the first character of the file name.
- After the images are collected, their features are extracted in batches. The program implements two techniques for feature extraction: flatted image and bag of visual words. The features will be stored as numpy array on disk.
- Optionally you can visualize the extracted features in 2-D space to get a rough idea of how successfully the features are extracted.
- Once the features are extracted and saved, you can train the classifier. The program only implements one classifier - Support Vector Machine. The classifier will also be saved on disk once the training completes.
- Lastly, the game can be played where the user will be prompted for random sign language of letters. The game is organized into ten rounds and is timed. The highest score is 10.

## Usage

**`usage: slr.py [-h] [--data-dir DATA_DIR] [--image-dir IMAGE_DIR] [--feature-type FEATURE_TYPE] [--model-file MODEL_FILE] {collect,extract,train,play,visualize}`**

- **`slr.py`** is the name of the Python script that is executed to run the program.
- **`[-h]`** is an optional argument that displays the help message when used.
- **`[--data-dir DATA_DIR]`** is an optional argument that specifies the directory where the program data is stored.
- **`[--image-dir IMAGE_DIR]`** is an optional argument that specifies the directory where the images are stored (under `DATA_DIR`).
- **`[--feature-type FEATURE_TYPE]`** is an optional argument that specifies the type of feature extraction method to be used. Currently, only `bow` and `flat` are supported.
- **`[--model-file MODEL_FILE]`** is an optional argument that specifies the file where the trained model is stored.
- **`{collect,extract,train,play,visualize}`** is a required argument that specifies the operation to be performed by the program. The options available are:
  - **`collect`** - collects data by capturing images of hand gestures and saving them to disk.
  - **`extract`** - extracts features from the images stored in the specified directory and saves them to disk.
  - **`train`** - trains a machine learning model using the extracted features.
  - **`play`** - uses the trained model to predict the hand gestures from a live video stream.
  - **`visualize`** - visualizes the feature distribution of the hand gesture images using t-SNE.
