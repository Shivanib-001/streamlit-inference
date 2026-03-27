# utils/model_yolov7.py

import cv2
import torch
import numpy as np

from utils.general import non_max_suppression, scale_coords
from utils.plots import colors
from utils.segment.general import process_mask


class YOLOv7Model:
    def __init__(self, model, device, names, img_size=(640, 640)):
        self.model = model
        self.device = device
        self.names = names
        self.img_size = img_size

    # ---------------- PREPROCESS ----------------
    def preprocess(self, frame):
        """
        Convert image to model input tensor
        """
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
        """
        Run model forward pass
        """
        with torch.no_grad():
            pred = self.model(img)
        return pred

    # ---------------- POSTPROCESS ----------------
    def postprocess(self, pred, img, original_frame, conf_thres=0.4, iou_thres=0.45):
        """
        Convert raw predictions to usable detections
        Supports BOTH detection and segmentation
        """

        proto = None

        # -------- HANDLE SEGMENTATION OUTPUT --------
        if isinstance(pred, (list, tuple)):
            pred, proto = pred[:2]

            # 🔥 unwrap proto safely
            while isinstance(proto, (list, tuple)):
                proto = proto[0]

        # -------- NMS --------
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            nm=32 if proto is not None else 0
        )

        results = []

        for i, det in enumerate(pred):

            if len(det) == 0:
                continue

            # Rescale boxes to original image
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], original_frame.shape
            ).round()

            boxes = []
            masks_out = []

            # -------- SEGMENTATION --------
            if proto is not None:

                if proto.ndim == 4:
                    proto_i = proto[i]
                else:
                    proto_i = proto

                masks = process_mask(
                    proto_i,
                    det[:, 6:],
                    det[:, :4],
                    img.shape[2:],
                    upsample=True
                )

                masks = masks.cpu().numpy()
                masks_out = masks

            # -------- BOXES --------
            for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                cls_id = int(cls)

                boxes.append({
                    "bbox": [int(x.item()) for x in xyxy],
                    "conf": float(conf),
                    "class_id": cls_id,
                    "class_name": self.names[cls_id],
                    "color": colors(cls_id, True)
                })

            results.append({
                "boxes": boxes,
                "masks": masks_out
            })

        return results

    # ---------------- FULL PIPELINE ----------------
    def predict(self, frame):
        """
        Complete pipeline:
        frame -> preprocess -> infer -> postprocess
        """
        img = self.preprocess(frame)
        pred = self.infer(img)
        results = self.postprocess(pred, img, frame)

        return results
