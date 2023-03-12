import cv2
import numpy as np
import pickle
from sklearn.preprocessing import scale

# Define the skin color ranges in HSV format
skin_lower = np.array([0, 20, 70], dtype=np.uint8)
skin_upper = np.array([20, 255, 255], dtype=np.uint8)

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale and reshape it to a vector
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(-1)

    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    image = scale(image)

    return image

# Define the recognize_gesture function
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Load the trained SVM model from a file
with open('svm_model.pickle', 'rb') as f:
    clf = pickle.load(f)
def recognize_gesture(image):
    # Make a prediction using the pre-trained SVM model
    prediction = clf.predict([image])
    label = labels[int(prediction)-1]
    return label


# Create a video capture object for the camera
camera = cv2.VideoCapture(0)

# Start the main loop
while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    if not ret:
        # If reading the frame failed, break out of the loop
        break

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask by thresholding the HSV image based on the skin color ranges
    mask = cv2.inRange(hsv, skin_lower, skin_upper)

    # Apply morphological operations to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find the contours of the binary mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour that meets certain criteria as the hand region
    max_area = 0
    hand_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if area > max_area and aspect_ratio > 0.5 and x > 0 and y > 0:
            max_area = area
            hand_contour = contour

    # Check if the hand region is found
    if hand_contour is not None:
        # Draw a bounding box around the hand region
        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the image to the bounding box of the hand region
        hand_image = frame[y:y + h, x:x + w]

        # Resize the cropped image to 28x28 pixels
        hand_image = cv2.resize(hand_image, (28, 28))

        # Convert the resized image to grayscale
        hand_image = preprocess_image(hand_image)

        # Pass the grayscale image to the gesture recognition function
        gesture = recognize_gesture(hand_image)
        print(gesture)

    # Show the original image with the hand region outlined
    cv2.imshow('Hand sign detection', frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
