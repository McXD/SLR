# requirements.txt
#
# joblib==1.1.0
# matplotlib==3.5.1
# numpy==1.21.5
# opencv_python==4.7.0.72
# pandas==1.4.2
# scikit_learn==1.2.2
# scipy==1.7.3
# seaborn==0.11.2

# usage: slr.py [-h] [--data-dir DATA_DIR] [--image-dir IMAGE_DIR] [--feature-type FEATURE_TYPE]
#               [--model-file MODEL_FILE]
#               {collect,extract,train,play,visualize}

import os
import sys
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
import string
import random
import time
from scipy.cluster.vq import vq
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd


def process_frame(frame):
    """
    Process a frame captured  by the camera
    """
    img_size = (128, 128)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, img_size)

    return gray


def collect_images(save_dir):
    """
    Collect images from camera
    """
    cap = cv2.VideoCapture(0)
    # Set the frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize counter for number of data files collected
    num_files = len(os.listdir(save_dir))

    while True:
        ret, frame = cap.read()

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('z'):
            break
        elif key >= ord('a') and key <= ord('z') and key != ord('j') and key != ord('z'):
            # Save image with corresponding label
            letter = chr(key).upper()
            filename = os.path.join(
                "data", "images", f"{letter}_{num_files}.jpg")

            # Process frame before saving
            gray = process_frame(frame)
            cv2.imwrite(filename, gray)

            num_files += 1

        # Display instructions on frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Press any letter key (excluding J and Z) to save an image",
                    (10, 50), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'z' to quit",
                    (10, 100), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{num_files} images collected",
                    (10, 150), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Camera", frame)

    cap.release()
    cv2.destroyAllWindows()


def extract_features_flat(img):
    """
    Flat the image pixels as feature vector
    """
    # Flatten image
    feature_vector = img.flatten()
    # Normalize feature vector
    feature_vector = normalize(feature_vector.reshape(1, -1), norm='l2')[0]

    return feature_vector


# Use global variable to store the codebook to avoid reloading
codebook = None


def extract_feature_bow(image):
    """
    Extract features using bag of visual words
    """
    global codebook
    codebook_file = f"{args.data_dir}/codebook.npy"

    if codebook is None and os.path.exists(codebook_file):
        # We have the codebook on disk, load the codebook from file
        codebook = np.load(codebook_file)
    elif not os.path.exists(codebook_file):
        # We don't the codebook on disk, build the codebook
        image_paths = [os.path.join('data/images', f)
                       for f in os.listdir('data/images')]
        features = []
        sift = cv2.SIFT_create()
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(img, None)
            features.append(descriptors)
        features = np.vstack(features)
        n_clusters = 30
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=100)
        kmeans.fit(features)
        codebook = kmeans.cluster_centers_
        np.save(codebook_file, codebook)

    # compute SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Compute the codebook ids
    codebook_ids, _ = vq(descriptors, codebook)

    # Compute the histogram of codebook ids
    histogram, _ = np.histogram(
        codebook_ids, bins=range(len(codebook) + 1), density=True)

    return histogram


extract_feature_funcs = {
    "bow": extract_feature_bow,
    "flat": extract_features_flat
}


def extract_features(image_dir, feature_type):
    """
    Extract features and labels from images
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    features = []
    labels = []

    # Loop through images and extract features
    for image_file in image_files:
        # Load image and resize to desired dimensions
        # Caution: needs to use grayscale for SIFT
        img = cv2.imread(os.path.join(image_dir, image_file),
                         cv2.IMREAD_GRAYSCALE)
        # Extract features with the specified feature type
        feature_vector = extract_feature_funcs[feature_type](img)

        # Add features and label to list
        features.append(feature_vector)
        # Convert letter to label (A=0, B=1, etc.)
        label = ord(image_file[0]) - ord('A')
        labels.append(label)

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Return features and labels
    return features, labels


def train_classifier(features, labels):
    """
    Train SVM classifier
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM classifier
    svm = SVC(kernel='linear', C=1, probability=True, verbose=1)
    svm.fit(X_train, y_train)

    # Make predictions on test set and calculate accuracy
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Return trained SVM classifier and accuracy
    return svm, accuracy


def play_game(clf, feature_type):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    # Define font for captions
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define round time limit in seconds
    round_time_limit = 20

    # All letters indexed by serial number
    letters = list(string.ascii_uppercase)

    # Start game loop
    score = 0
    round_num = 1
    while True:
        letter = random.choice(letters)
        prompt = "Round " + str(round_num) + " - Letter: " + letter
        score_caption = "Score: " + str(score)

        # Reset round timer
        round_start_time = time.time()

        # Enter round loop
        round_correct = False
        while time.time() - round_start_time < round_time_limit:
            # Read frame from camera
            ret, frame_capped = cap.read()

            # Preprocess frame
            frame = process_frame(frame_capped)

            try:
                # Extract features from frame
                features = extract_feature_funcs[feature_type](frame)

                # Make prediction using SVM model
                label = clf.predict([features])[0]

                # Get corresponding letter for label
                predicted_letter = letters[label]

                # Check if predicted letter is correct
                if predicted_letter == letter:
                    round_correct = True
                    break

                # Captions
                cv2.putText(frame_capped, prompt, (10, 50),
                            font, 0.6, (255, 255, 255), 1)
                cv2.putText(frame_capped, score_caption, (10, 100),
                            font, 0.6, (255, 255, 255), 1)
                cv2.putText(frame_capped, "Predicted: " + predicted_letter,
                            (10, 150), font, 0.6, (0, 0, 255), 1)
                time_caption = "Round time left: {:.0f}s".format(
                    round(round_time_limit - (time.time() - round_start_time)))
                cv2.putText(frame_capped, time_caption, (10, 200),
                            font, 0.6, (0, 0, 255), 1)

                # Show frame
                cv2.imshow("Game", frame_capped)

                # Check for key press
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            except Exception as e:
                print(e, file=sys.stderr)

        # Update score and round number
        if round_correct:
            score += 1

        round_num += 1

        # Check if game is over
        if round_num > 10:
            break

    # Print final score and release camera
    print("Final score:", score)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Sign Language Recognition')
    parser.add_argument('mode', choices=[
                        'collect', 'extract', 'train', 'play', 'visualize'], help='Mode of operation')
    parser.add_argument('--data-dir', default='data',
                        help='Directory for storing data')
    parser.add_argument('--image-dir', default='images',
                        help='Directory for storing collected images')
    parser.add_argument('--feature-type', default='flat',
                        help='Type of feature extraction to use (flat, bow)')
    parser.add_argument('--model-file', default='model.pkl',
                        help='File for storing trained model')
    args = parser.parse_args()

    # Check if data directory exists, create it if not
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Perform action based on selected mode
    if args.mode == 'collect':
        # Collect images
        collect_images(os.path.join(args.data_dir, args.image_dir))
    elif args.mode == 'extract':
        # Extract features
        features, labels = extract_features(os.path.join(
            args.data_dir, args.image_dir), args.feature_type)
        # Save features and labels to disk
        np.save(os.path.join(args.data_dir,
                f'{args.feature_type}_features.npy'), features)
        np.save(os.path.join(args.data_dir,
                f'{args.feature_type}_labels.npy'), labels)
    elif args.mode == 'visualize':
        # Load features and labels
        features = np.load(os.path.join(args.data_dir,
                                        f'{args.feature_type}_features.npy'))
        labels = np.load(os.path.join(args.data_dir,
                                      f'{args.feature_type}_labels.npy'))

        # Reduce feature dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)

        # Visualize using Seaborn
        df = pd.DataFrame(features_tsne, columns=['x', 'y'])
        df['label'] = [chr(ord('A') + label_id) for label_id in labels]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='label',
                        palette='tab10', ax=ax)

        plt.show()
    elif args.mode == 'train':
        # Extract features from collected images
        features = np.load(os.path.join(args.data_dir,
                                        f'{args.feature_type}_features.npy'))
        labels = np.load(os.path.join(args.data_dir,
                                      f'{args.feature_type}_labels.npy'))
        # Train SVM classifier
        svm, accuracy = train_classifier(features, labels)
        print(
            f'Trained SVM classifier with {len(features)} samples and accuracy {accuracy:.2f}')

        # Save trained SVM classifier to file
        model_file = os.path.join(args.data_dir, args.model_file)
        joblib.dump(svm, model_file)
        print(f'Saved SVM classifier to {model_file}')
    elif args.mode == 'play':
        # Load classifier from disk
        clf = joblib.load(os.path.join(args.data_dir, args.model_file))
        # Play game
        play_game(clf, args.feature_type)
    else:
        print('Invalid mode selected')
