import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
from detectron2.structures import Boxes, Instances

def plate_finder():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.DEVICE = "cpu"  

    predictor = DefaultPredictor(cfg)
    image = cv2.imread('C:\\Users\\nabil\\Downloads\\calorie-pic-tracker\\calorie-pic-tracker\\plateAndBowlFinder\\plate.jpg')

    if image is None:
        print("Error: Image not found or cannot be loaded.")
        exit()

    outputs = predictor(image)
    classes = outputs["instances"].pred_classes
    bowl_class_id = 44
    bowl_indices = (classes == bowl_class_id).nonzero(as_tuple=True)[0]

    # Create an empty Instances object if no bowls are detected
    if len(bowl_indices) > 0:
        bowls = outputs["instances"][bowl_indices]
    else:
        bowls = Instances(image.shape[:2])
        bowls.pred_boxes = Boxes(torch.empty((0, 4)))
        bowls.scores = torch.empty((0, ))
        bowls.pred_classes = torch.empty((0,), dtype=torch.int64)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)  # Set scale to 1.0 or adjust as needed
    out = v.draw_instance_predictions(bowls.to("cpu"))

    # Resize the image for display
    max_width = 800  # Fixed width for display
    max_height = 600  # Fixed height for display
    resize_factor = min(max_width / out.get_image().shape[1], max_height / out.get_image().shape[0])
    resized_image = cv2.resize(out.get_image(), (int(out.get_image().shape[1] * resize_factor), int(out.get_image().shape[0] * resize_factor)))

    # Display the resized image
    cv2.imshow("Bowl Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(plate_finder())
