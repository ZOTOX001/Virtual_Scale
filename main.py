import cv2
import numpy as np

# Constants
KNOWN_OBJECT_LENGTH_CM = 15  # Known length of the object in centimeters (adjust to real-world size)
pixels_per_cm = None  # Placeholder for pixels per centimeter

def calibrate_pixels_per_cm(frame):
    """
    Assume the object size is known (e.g., 15 cm). 
    We calculate pixels per cm by using the object's length as reference.
    The user needs to place an object of known size in the frame.
    """
    global pixels_per_cm

    # Convert the frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to isolate the object
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 2000]  # Filter out small contours

    if contours:
        # Find the largest contour (assumed to be the object of interest)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the pixels-per-centimeter based on the known size of the object
        pixels_per_cm = w / KNOWN_OBJECT_LENGTH_CM  # Using the width of the bounding box for calibration
        print(f"Calibration successful! Pixels per cm: {pixels_per_cm:.2f}")
    else:
        print("Error: No suitable object detected for calibration.")


def process_frame(frame):
    global pixels_per_cm

    if pixels_per_cm is None:
        return frame  # Skip processing if calibration hasn't been done

    # Convert the frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to detect the object
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 2000]  # Adjust area threshold

    if contours:
        # Find the largest contour (assumed to be the object of interest)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box around the object
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate object dimensions in centimeters
        obj_length_cm = w / pixels_per_cm  # The width of the bounding box is the object length

        # Draw bounding box and display only the length
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Length: {obj_length_cm:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No object detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'c' to calibrate using an object of known size. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Display instructions before calibration
    if pixels_per_cm is None:
        cv2.putText(frame, "Place object and press 'c' to calibrate", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Process the frame for measurement
    processed_frame = process_frame(frame)

    # Show the processed frame
    cv2.imshow("Object Measurement", processed_frame)

    # Check for keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Calibrate when 'c' is pressed
        calibrate_pixels_per_cm(frame)
    elif key == ord('q'):  # Quit when 'q' is pressed
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
