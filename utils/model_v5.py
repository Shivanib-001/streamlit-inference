# utils/model_yolov5.py

import cv2
import torch
import numpy as np


class YOLOv5Model:
    def __init__(self, model, device, names=None, img_size=(640, 640)):
        self.model = model
        self.device = device
        self.img_size = img_size

        # If names not passed, try from model
        if names is None:
            try:
                self.names = model.names
            except:
                self.names = {}
        else:
            self.names = names

    # ---------------- PREPROCESS ----------------
    def preprocess(self, frame):
        img = cv2.resize(frame, self.img_size)
        img = img[:, :, ::-1]  # BGR → RGB
        return img

    # ---------------- PREDICT ----------------
    def predict(self, frame):

        try:
            img = self.preprocess(frame)

            # 🔥 NEW ULTRALYTICS STYLE
            results = self.model(img)

            # results[0] → first image
            res = results[0]

            boxes = []

            if hasattr(res, "boxes"):  # ✅ Ultralytics format

                for box in res.boxes:

                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    if cls_id >= len(self.names):
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)

                    boxes.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": self.names[cls_id],
                        "color": (0, 255, 0)
                    })

            else:
                print("[ERROR] Unknown model output format")
                return []

            return [{
                "boxes": boxes,
                "masks": []
            }]

        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return []
