import mediapipe as mp
import cv2
import math


def distanceFinder(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def sizeOfPixels(image, handLength):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode = True, max_num_hands = 1, min_detection_confidence = 0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Load an image of a hand
    image = cv2.imread(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

            middle_finger_id = 12
            palm_id = 0

            middle_finger_tip = hand_landmarks.landmark[middle_finger_id]
            palm_base = hand_landmarks.landmark[palm_id]

            h, w, _ = image.shape
            middle_finger_tip_pixel = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))
            palm_base_pixel = (int(palm_base.x * w), int(palm_base.y * h))

            distance = distanceFinder(palm_base_pixel, middle_finger_tip_pixel)

            pixelSize = distance / handLength

            return pixelSize, distance
        
image = cv2.imread("calorie-pic-tracker/handFinder/20240905_142634.jpg")

print(sizeOfPixels("calorie-pic-tracker/handFinder/20240905_142634.jpg", 19))
