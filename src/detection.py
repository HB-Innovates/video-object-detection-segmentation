class YOLOv5Detector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        import torch
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)

    def detect_objects(self, image):
        results = self.model(image)
        return results

    def draw_detections(self, image, results):
        import cv2
        annotated_image = results.render()[0]
        return annotated_image