# utils/model_factory.py

from pathlib import Path

# Import model handlers
from utils.model_yolov7 import YOLOv7Model

# (future)
from utils.model_yolov5 import YOLOv5Model
# from utils.model_yolov8 import YOLOv8Model


class ModelFactory:

    @staticmethod
    def detect_model_type(model_path):
        """
        Detect YOLO version from filename
        """
        name = Path(model_path).name.lower()

        if "yolov7" in name:
            return "yolov7"
        elif "yolov5" in name:
            return "yolov5"
        elif "yolov8" in name:
            return "yolov8"
        else:
            return "unknown"

    @staticmethod
    def is_segmentation(model_path):
        """
        Detect if segmentation model
        """
        name = Path(model_path).name.lower()
        return "seg" in name

    @staticmethod
    def create(model, device, names, model_path):
        """
        Factory method to return correct model handler
        """

        model_type = ModelFactory.detect_model_type(model_path)
        is_seg = ModelFactory.is_segmentation(model_path)

        print(f"[Factory] Detected model type: {model_type}")
        print(f"[Factory] Segmentation: {is_seg}")

        # -------- YOLOv7 --------
        if model_type == "yolov7":
            return YOLOv7Model(model, device, names)

        # -------- YOLOv5 (future) --------
        elif model_type == "yolov5":
            return YOLOv5Model(model, device, names)

        # -------- YOLOv8 (future) --------
        elif model_type == "yolov8":
            raise NotImplementedError("YOLOv8 handler not added yet")

        # -------- UNKNOWN --------
        else:
            raise ValueError(f"Unsupported model type: {model_path}")
            
