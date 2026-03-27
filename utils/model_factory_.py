# utils/model_factory.py

import torch


class ModelFactory:

    @staticmethod
    def create(model_path, device="cpu"):
        print(model_path)
        # -------- TRY ULTRALYTICS FIRST --------
        try:
            from ultralytics import YOLO

            print("[INFO] Trying Ultralytics loader...")
            model = YOLO(model_path)

            print("[SUCCESS] Ultralytics model loaded")

            from utils.model_yolov5 import YOLOv5Model
            return YOLOv5Model(model, device)

        except Exception as e:
            print("[ERROR] Ultralytics load failed:")
            print(e)

        # -------- TRY TORCH LOAD --------
        try:
            print("[INFO] Trying torch.load...")

            model = torch.load(model_path, map_location=device, weights_only=False)

            if isinstance(model, dict):
                model = model.get('model', model)

            model.to(device).eval()

            print("[SUCCESS] Torch model loaded")

            from utils.model_yolov5_legacy import YOLOv5LegacyModel
            return YOLOv5LegacyModel(model, device)

        except Exception as e:
            print("[ERROR] torch.load failed:")
            print(e)

        # -------- FINAL --------
        raise Exception("Unsupported model format")
