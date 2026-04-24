import onnxruntime as ort
import cv2
import numpy as np
import torch
#from torchvision.ops import nms

MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
INPUT_SIZE = 550

class YOLACTHandler:
    def __init__(self, model_path, providers=None):
        
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
        
        model_path=model_path
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.priors=self.generate_priors()
        print(self.input_name, self.output_names, model_path)

        # change as needed
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
               
               
    # ---------------- PREPROCESS ----------------
    def preprocess(self, img):
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        img = (img - MEANS) / STD
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, 0).astype(np.float32)
        return img

    def generate_priors(self):
        feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]
        aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
        scales = [24, 48, 96, 192, 384]
        priors = []
        for idx, fsize in enumerate(feature_map_sizes):
            scale = scales[idx]
            for y in range(fsize[0]):
                for x in range(fsize[1]):
                    cx = (x + 0.5) / fsize[1]
                    cy = (y + 0.5) / fsize[0]
                    for ratio in aspect_ratios[idx]:
                        r = np.sqrt(ratio)
                        w = scale / INPUT_SIZE * r
                        h = scale / INPUT_SIZE / r
                        priors.append([cx, cy, w, h])
        return np.array(priors, dtype=np.float32)

    def decode(self,loc, priors, variances=[0.1, 0.2]):
        boxes = np.zeros_like(loc)
        boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
        boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def nms(self,boxes, scores, iou_threshold=0.5):
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )
        return np.array(indices).flatten() if len(indices) > 0 else np.array([], dtype=int)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))



    # ---------------- POSTPROCESS ----------------
    def postprocess(self, output, original_shape, conf_thres=0.1, iou_thres=0.5):

        loc, conf, mask, _, proto = output

        loc = np.squeeze(loc, axis=0)
        conf = np.squeeze(conf, axis=0)
        mask = np.squeeze(mask, axis=0)
        proto = np.squeeze(proto, axis=0)

        # -------- CLASS + SCORE --------
        scores = np.max(conf[:, 1:], axis=1)
        classes = np.argmax(conf[:, 1:], axis=1)

        keep = scores > conf_thres
        if not np.any(keep):
            return []

        scores = scores[keep]
        classes = classes[keep]
        mask = mask[keep]
        loc = loc[keep]
        priors = self.priors[keep]

        # -------- DECODE BOXES --------
        boxes = self.decode(loc, priors)

        # -------- NMS --------
        keep_nms = self.nms(boxes, scores, iou_threshold=iou_thres)

        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        classes = classes[keep_nms]
        mask = mask[keep_nms]

        # -------- MASK GENERATION --------
        masks = proto @ mask.T
        masks = self.sigmoid(masks)
        masks = np.transpose(masks, (2, 0, 1))

        resized_masks = []
        
        h, w = original_shape

        for m in masks:
            resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(resized > 0.5)

        resized_masks = np.array(resized_masks, dtype=bool)

        # -------- FORMAT OUTPUT --------
        results = []

        boxes_out = []

        for i in range(len(boxes)):
            bbox = boxes[i]

            # scale to image size
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)

            cls_id = int(classes[i])
            score = float(scores[i])

            boxes_out.append({
                "bbox": [x1, y1, x2, y2],
                "conf": score,
                "class_id": cls_id,
                "class_name": self.names[cls_id],
                "color": (0, 255, 0)  
            })

        results.append({
            "boxes": boxes_out,
            "masks": []
        })

        return results


    # ---------------- MAIN PREDICT ----------------
    def predict(self, frame):
        inp = self.preprocess(frame)
        self
        outputs = self.session.run(self.output_names, {self.input_name: inp})

        results = self.postprocess(outputs, (frame.shape[:2]))
        return results
