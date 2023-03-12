import cv2
import numpy as np
import os
from skimage.feature import hog

# Set parameters for HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)

# Set path to directory containing images
data_path = "data/images"

# Initialize list to store features and labels
features = []
labels = []

# Loop over all images in directory
for filename in os.listdir(data_path):
    # Load image
    img = cv2.imread(os.path.join(data_path, filename))
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize image to (64, 64)
    resized = cv2.resize(gray, (64, 64))
    # Extract HOG features from image
    hog_features = hog(resized, orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       transform_sqrt=True, block_norm="L1")
    # Extract label from filename
    label = ord(filename[0].upper()) - ord('A')
    # Append features and label to lists
    features.append(hog_features)
    labels.append(label)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Print shape of features and labels arrays
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Save features and labels to disk
np.save("data/features_hog.npy", features)
np.save("data/labels_hog.npy", labels)
