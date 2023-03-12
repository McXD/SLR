import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# Set parameters for bag of visual words feature extraction
num_clusters = 128
vectorizer = CountVectorizer()

# Set path to directory containing images
data_path = "data/images"

# Initialize list to store SIFT descriptors
descriptors = []

# Loop over all images in directory
for filename in os.listdir(data_path):
    # Load image
    img = cv2.imread(os.path.join(data_path, filename))
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize image to (128, 128)
    resized = cv2.resize(gray, (128, 128))
    # Extract SIFT features from image
    sift = cv2.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(resized, None)
    # Append descriptors to list of SIFT descriptors
    descriptors.extend(descriptor)

# Convert list of descriptors to numpy array
descriptors = np.array(descriptors)

# Train k-means clustering model on the SIFT descriptors to obtain a codebook
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(descriptors)

# Convert the codebook to a bag of visual words dictionary
codebook = vectorizer.fit_transform(
    [" ".join([str(x) for x in cluster]) for cluster in kmeans.cluster_centers_])

# Save k-means clustering model and bag of visual words dictionary to disk
np.save("data/kmeans.npy", kmeans)
np.save("data/codebook.npy", codebook)

# Define function for extracting bag of visual words features from an image

def extract_features(image_path):
    # Load image
    img = cv2.imread(image_path)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize image to (128, 128)
    resized = cv2.resize(gray, (128, 128))
    # Extract SIFT features from image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(resized, None)
    # Assign each feature to the nearest codebook cluster
    clusters = kmeans.predict(descriptors)
    # Convert the cluster assignments to a bag of visual words histogram
    histogram = vectorizer.transform([" ".join([str(x) for x in clusters])])
    # Convert the histogram to a 1D array
    feature_vector = histogram.toarray()[0]
    # Normalize the feature vector
    feature_vector = normalize(feature_vector.reshape(1, -1), norm='l2').reshape(-1,)

    return feature_vector


# Loop over all images in directory again to extract features and save them to disk
features = []
labels = []
for filename in os.listdir(data_path):
    # Extract label from filename
    label = ord(filename[0].upper()) - ord('A')
    # Extract features from image and add to list of features
    feature_vector = extract_features(os.path.join(data_path, filename))
    features.append(feature_vector)
    labels.append(label)

# Convert list of features to numpy array
features = np.array(features)

# Save features and labels to disk
np.save("data/features_bovw.npy", features)
np.save("data/labels_bovw.npy", labels)
