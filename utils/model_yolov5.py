# utils/model_yolov5.py

import cv2
import torch
import numpy as np

from utils.general import non_max_suppression, scale_coords
from utils.plots import colors


class YOLOv5Model:
    def __init__(self, model, device, names, img_size=(640, 640)):
        self.model = model
        self.device = device
        self.names = names
        self.img_size = img_size

    # ---------------- PREPROCESS ----------------
    def preprocess(self, frame):
        img = cv2.resize(frame, self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255.0

        if img.ndim == 3:
            img = img.unsqueeze(0)

        return img

    # ---------------- INFERENCE ----------------
    def infer(self, img):
        with torch.no_grad():
            pred = self.model(img)
        return pred

    # ---------------- POSTPROCESS ----------------
    def postprocess(self, pred, img, original_frame,
                    conf_thres=0.4, iou_thres=0.45):

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        results = []

        for i, det in enumerate(pred):

            if det is None or len(det) == 0:
                continue

            # Rescale boxes to original image
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], original_frame.shape
            ).round()

            boxes = []

            for row in det:

                # -------- SAFE EXTRACTION --------
                x1, y1, x2, y2 = row[:4]

                # Confidence (safe default)
                conf = float(row[4])

                # 🔥 ALWAYS take last value as class id
                cls_id = int(row[-1])

                # -------- VALIDATION --------
                if cls_id < 0 or cls_id >= len(self.names):
                    print(f"[WARN] Skipping invalid class id: {cls_id}")
                    continue

                boxes.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": self.names[cls_id],
                    "color": colors(cls_id, True)
                })

            results.append({
                "boxes": boxes,
                "masks": []   
            })

        return results

    # ---------------- FULL PIPELINE ----------------
    def predict(self, frame):
        try:
            img = self.preprocess(frame)
            pred = self.infer(img)
            results = self.postprocess(pred, img, frame)
            return results

        except Exception as e:
            print(f"[ERROR] YOLOv5 prediction failed: {e}")
            return []
