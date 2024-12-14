import cv2
import numpy as np
from time import sleep

# Minimum width and height of the detected objects to be considered as valid vehicles
largura_min = 80  # Minimum object width
altura_min = 80   # Minimum object height

# Tolerance offset for the line crossing detection
offset = 6

# Position of the line to detect vehicle crossing
pos_linha = 550

# Frames per second (FPS) for video playback
delay = 60

# List to store detected vehicle centers
detec = []

# Counter for the number of vehicles detected
carros = 0

# Function to calculate the center of a rectangle
def pega_centro(x, y, w, h):
    x1 = int(w / 2)  # Half of the width
    y1 = int(h / 2)  # Half of the height
    cx = x + x1      # X-coordinate of the center
    cy = y + y1      # Y-coordinate of the center
    return cx, cy

# Load the video file
cap = cv2.VideoCapture('video.mp4')

# Background subtraction for detecting moving objects
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    # Read a frame from the video
    ret, frame1 = cap.read()

    # Calculate delay for smooth playback
    tempo = float(1 / delay)
    sleep(tempo)

    # Convert the frame to grayscale for processing
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction to detect moving objects
    img_sub = subtracao.apply(blur)

    # Dilate the image to fill small gaps in detected objects
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply morphological closing to smoothen the binary image
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Find contours in the processed image
    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the detection line on the original frame
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (176, 130, 39), 2)

    # Loop through all detected contours
    for (i, c) in enumerate(contorno):
        # Get the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Validate if the contour meets the minimum size criteria
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        # Draw a rectangle around the detected object
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the center of the detected object
        centro = pega_centro(x, y, w, h)

        # Add the center to the list of detections
        detec.append(centro)

        # Draw a circle at the center of the detected object
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        # Check if the center crosses the detection line
        for (x, y) in detec:
            if (y < (pos_linha + offset)) and (y > (pos_linha - offset)):
                # Increment vehicle count if crossing is detected
                carros += 1
                # Change the color of the line to indicate detection
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                # Remove the center from the list to avoid recounting
                detec.remove((x, y))
                print("No. of cars detected: " + str(carros))

    # Display the total vehicle count on the frame
    cv2.putText(frame1, "VEHICLE COUNT: " + str(carros), (320, 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

    # Show the original video with detected vehicles
    cv2.imshow("Video Original", frame1)

    # Show the processed binary image used for detection
    cv2.imshow("Detection", dilatada)

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources and close all windows
cv2.destroyAllWindows()
cap.release()
