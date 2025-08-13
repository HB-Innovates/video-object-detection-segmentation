import logging

class YOLOv5Detector:
    def __init__(self, model_path):
        """
        Initialize YOLOv5 detector.
        Args:
            model_path (str): Path to YOLOv5 weights.
        """
        self.model_path = model_path
        self.model = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        import torch
        self.logger.info(f"Loading YOLOv5 model from {self.model_path}")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        self.logger.info("Model loaded successfully.")

    def detect_objects(self, image):
        self.logger.info("Running object detection...")
        results = self.model(image)
        self.logger.info(f"Detection complete. Found {len(results.xyxy[0])} objects.")
        return results

    def draw_detections(self, image, results):
        import cv2
        self.logger.info("Drawing detections on image.")
        annotated_image = results.render()[0]
        return annotated_image