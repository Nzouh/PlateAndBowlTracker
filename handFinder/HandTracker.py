import mediapipe as mp
import cv2
import math
import numpy as np

def distanceFinder(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def findPixelSize(pathToFile, handLength):
    # Initialize MediaPipe Hand Tracking and Drawing utils
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Read and convert the image to RGB
    hand_picture = cv2.imread(pathToFile)
    image_rgb = cv2.cvtColor(hand_picture, cv2.COLOR_BGR2RGB)

    # Process the image to find hand landmarks
    results = hands.process(image_rgb)

    # Get original image dimensions
    original_h, original_w, _ = hand_picture.shape
    print(f"Image dimensions: {original_w} x {original_h} pixels")

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to pixel coordinates using original image dimensions
            base = hand_landmarks.landmark[0]  # wrist landmark
            tip = hand_landmarks.landmark[12]  # middle finger tip

            # Calculate pixel coordinates for the original image
            base_coords_original = (int(base.x * original_w), int(base.y * original_h))
            tip_coords_original = (int(tip.x * original_w), int(tip.y * original_h))

            distanceOfHandInPixel = distanceFinder(base_coords_original, tip_coords_original)
            
            pixelSize = distanceOfHandInPixel / handLength

            # Draw landmarks on the original image for confirmation
            cv2.circle(hand_picture, base_coords_original, 5, (0, 255, 0), -1)
            cv2.circle(hand_picture, tip_coords_original, 5, (0, 255, 0), -1)

            # Calculate Euclidean distance (hand length in pixels) in the original image
            hand_length_px = np.linalg.norm(np.array(base_coords_original) - np.array(tip_coords_original))
            print(f"Hand length in pixels (original image): {hand_length_px}")

            #Calculate the Ratio 

    # Now resize the image for display
    max_width = 1920  # Adjust this value to control resizing
    max_height = 900  # Optional height adjustment

    # Calculate scale factors based on width and height
    scale_factor_w = max_width / original_w
    scale_factor_h = max_height / original_h

    # Use the smaller scaling factor to maintain aspect ratio
    scale_factor = min(scale_factor_w, scale_factor_h)

    # Resize the image using the calculated scale factor
    # resized_image = cv2.resize(hand_picture, (int(original_w * scale_factor), int(original_h * scale_factor)))

    # Display the resized image with landmarks
    # cv2.imshow('Hand Tracking', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return pixelSize

print(findPixelSize("calorie-pic-tracker/handFinder/20240905_142634.jpg", 19))
