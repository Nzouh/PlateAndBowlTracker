import cv2
import mediapipe as mp
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from handFinder import sizeOfPixels

hand_length_cm = 19


# Calculate distance between two points
def distanceFinder(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Detect hand and calculate pixel-to-cm ratio

# Detect plate using Detectron2
def find_plate_diameter(image, predictor):
    outputs = predictor(image)
    classes = outputs["instances"].pred_classes
    bowl_class_id = 44  # Class ID for bowls
    # Filter for bowl class
    bowl_indices = (classes == bowl_class_id).nonzero(as_tuple=True)[0]
    if len(bowl_indices) > 0:
        bowls = outputs["instances"][bowl_indices]
        # Extract bounding box of the bowl
        bbox = bowls.pred_boxes.tensor[0].numpy()
        x1, y1, x2, y2 = bbox
        diameter_px = x2 - x1
        return diameter_px
    print("wtf man")
    return None

# Main function to integrate hand and plate detection
def measure_plate(image_path, hand_length_cm):
    print("hello")
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        exit()


    # Step 1: Hand detection to calculate pixel-to-cm ratcio
    pixel_to_cm = sizeOfPixels(image, hand_length_cm) / hand_length_cm
    if pixel_to_cm is None:
        print("Error: Unable to detect hand.")
        return

    # Step 2: Detect plate using Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    plate_diameter_px = find_plate_diameter("calorie-pic-tracker/plateAndBowlFinder/plate.jpg", predictor)
    if plate_diameter_px is None:
        print("Error: Unable to detect plate.")
        return

    # Step 3: Convert plate diameter from pixels to cm
    plate_diameter_cm = plate_diameter_px / pixel_to_cm
    print(f"Plate diameter: {plate_diameter_cm:.2f} cm")

    # Optional: Display the detected hand and plate
    cv2.imshow("Hand and Plate Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("EHllo")
# Example usage
measure_plate("calorie-pic-tracker/handFinder/20240905_142634.jpg", hand_length_cm=19)
