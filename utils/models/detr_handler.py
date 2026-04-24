import onnxruntime as ort
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.ops import nms

class DETRHandler:
    def __init__(self, model_path, providers=None):
        print(model_path,providers)
        if providers is None:
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "./trt_cache",
                    },
                ),
                "CUDAExecutionProvider"
            ]
        print(model_path,providers)
        model_path=model_path
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        # change as needed
        self.names = ['NA', 'Trunk', 'NA']

    # ---------------- PREPROCESS ----------------
    def preprocess(self, frame):
        resized_frame = cv2.resize(frame, (1280, 720))
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_t = self.transform(img_rgb).unsqueeze(0).numpy()
        return img_t

    # ---------------- POSTPROCESS ----------------
    def postprocess(self, outputs, frame, score_thresh=0.1, nms_thresh=0.45):
        h, w = frame.shape[:2]

        pred_logits, pred_boxes = outputs[:2]

        probs = torch.softmax(torch.from_numpy(pred_logits[0]), -1)
        scores, labels = probs.max(-1)
        keep = scores > score_thresh

        boxes, scores_keep, labels_keep = [], [], []

        for box, label, score in zip(pred_boxes[0][keep], labels[keep], scores[keep]):
            bbox = self.rescale_bboxes(box, (w, h))
            boxes.append(bbox)
            scores_keep.append(score.item())
            labels_keep.append(label.item())

        results = []

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores_keep = torch.tensor(scores_keep)
            labels_keep = torch.tensor(labels_keep)

            keep_idx = nms(boxes, scores_keep, nms_thresh)

            det = {"boxes": [], "masks": []}

            for i in keep_idx:
                bbox = boxes[i].int().numpy()
                cls_id = labels_keep[i].item()
                score = scores_keep[i].item()

                cls_name = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)

                det["boxes"].append({
                    "bbox": bbox,
                    "conf": score,
                    "class_name": cls_name,
                    "color": (0, 255, 0)
                })
            print(det["boxes"])
            results.append(det)

        return results

    def rescale_bboxes(self, box, size):
        img_w, img_h = size
        return np.array([
            (box[0] - box[2]/2) * img_w,
            (box[1] - box[3]/2) * img_h,
            (box[0] + box[2]/2) * img_w,
            (box[1] + box[3]/2) * img_h,
        ])

    # ---------------- MAIN PREDICT ----------------
    def predict(self, frame):
        inp = self.preprocess(frame)

        outputs = self.session.run(
            ["pred_logits", "pred_boxes"],
            {"images": inp}
        )

        results = self.postprocess(outputs, frame)
        return results
