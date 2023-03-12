import cv2
import numpy as np
import pickle
from sklearn.preprocessing import scale

# Load the trained SVM model from a file
with open('svm_model.pickle', 'rb') as f:
    clf = pickle.load(f)

# Define a function to preprocess the image


def preprocess_image(image):
    # Convert the image to grayscale and resize it to 28x28 pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))

    # Reshape the image to a vector
    image = image.reshape(-1)

    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    image = scale(image)

    return image

# Define the main function


def main():
    # Create a video capture object for the camera
    camera = cv2.VideoCapture(0)

    # Start the main loop
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()

        if not ret:
            # If reading the frame failed, break out of the loop
            break

        # Preprocess the image
        image = preprocess_image(frame)

        # Make a prediction using the pre-trained SVM model
        prediction = clf.predict([image])

        # Convert the prediction to a letter label
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        label = labels[int(prediction) - 1]

        # Draw the predicted letter label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (50, 50), font,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame on the screen
        cv2.imshow('Hand sign detection', frame)

        # Check if the user wants to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    camera.release()
    cv2.destroyAllWindows()


# Call the main function
if __name__ == '__main__':
    main()
