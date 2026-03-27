# app.py

import streamlit as st
import cv2
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from collections import deque, Counter

# YOLO
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Factory
from utils.model_factory import ModelFactory

# ---------------- CONFIG ----------------
st.set_page_config(page_title="YOLO AI Dashboard", layout="wide")

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "runs"

# ---------------- MODEL MANAGER ----------------
class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

    def list_models(self):
        return [f.name for f in self.model_dir.glob("*.pt")]

    def load(self, model_name):
        path = self.model_dir / model_name

        device = select_device('0' if torch.cuda.is_available() else 'cpu')

        model = DetectMultiBackend(
            path,
            device=device,
            fp16=True
        )

        names = model.names

        return model, device, names, str(path)

# ---------------- GPU STATS ----------------
def get_gpu_stats():
    if torch.cuda.is_available():
        try:
            import subprocess
            result = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,nounits,noheader"]
            ).decode()

            util, used, total = result.strip().split(',')
            return int(util), int(used), int(total)
        except:
            return 0, 0, 0
    return 0, 0, 0

# ---------------- RENDER ----------------
def render(frame, results):
    """
    Draw boxes + masks
    """
    for det in results:
        boxes = det["boxes"]
        masks = det["masks"]

        # ---- Masks ----
        if len(masks) > 0:
            for i, mask in enumerate(masks):
                color = boxes[i]["color"]
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask > 0] = color
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.4, 0)

        # ---- Boxes ----
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            conf = box["conf"]
            cls = box["class_name"]
            color = box["color"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return frame

# ---------------- INIT ----------------
manager = ModelManager(MODEL_DIR)

fps_hist = deque(maxlen=30)
lat_hist = deque(maxlen=30)
conf_hist = deque(maxlen=100)

# ---------------- UI ----------------
st.title("AI Detection Dashboard")

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Controls")

    model_list = manager.list_models()

    if not model_list:
        st.warning("No models found in /models")
        st.stop()

    selected_model = st.selectbox("Select Model", model_list)

    source = st.radio("Input Source", ["Camera", "Video File"])

    video_file = None
    if source == "Video File":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    run = st.toggle("Start")

    st.markdown("---")
    st.subheader("📊 Performance")

    fps_box = st.empty()
    lat_box = st.empty()
    cpu_box = st.empty()
    gpu_box = st.empty()
    vram_box = st.empty()
    conf_box = st.empty()

    st.markdown("---")
    st.subheader("📦 Detections")

    total_box = st.empty()
    class_box = st.empty()

with col1:
    frame_window = st.empty()

# ---------------- LOAD MODEL ----------------
if run and "handler" not in st.session_state:

    with st.spinner("Loading model..."):
        model, device, names, path = manager.load(selected_model)

        handler = ModelFactory.create(
            model,
            device,
            names,
            path
        )

        st.session_state["handler"] = handler

def get_camera():
    for i in range(5):  # try 0–4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            return cap
    return None




# ---------------- VIDEO SOURCE ----------------
if run:

    handler = st.session_state["handler"]



    if source == "Camera":
        cap = get_camera()
        if cap is None:
            st.error("❌ No camera found")
            st.stop()

    else:
        tfile = Path("temp_video.mp4")
        with open(tfile, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(str(tfile))

    prev_time = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # -------- INFERENCE --------
        start = time.time()
        results = handler.predict(frame)
        latency = (time.time() - start) * 1000

        # -------- RENDER --------
        frame = render(frame, results)

        # -------- FPS --------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        fps_hist.append(fps)
        lat_hist.append(latency)

        # -------- STATS --------
        classes = []
        confs = []

        for det in results:
            for b in det["boxes"]:
                classes.append(b["class_name"])
                confs.append(b["conf"])

        if confs:
            conf_hist.extend(confs)

        avg_fps = sum(fps_hist) / len(fps_hist)
        avg_lat = sum(lat_hist) / len(lat_hist)
        avg_conf = sum(conf_hist) / len(conf_hist) if conf_hist else 0

        cpu = psutil.cpu_percent()
        gpu, used, total = get_gpu_stats()

        counter = Counter(classes)

        # -------- UI --------
        frame_window.image(frame, channels="BGR", width=True)

        fps_box.metric("FPS", f"{avg_fps:.2f}")
        lat_box.metric("Latency (ms)", f"{avg_lat:.1f}")
        cpu_box.metric("CPU %", cpu)
        gpu_box.metric("GPU %", gpu)
        vram_box.metric("VRAM (MB)", f"{used}/{total}")
        conf_box.metric("Avg Confidence", f"{avg_conf:.2f}")

        total_box.metric("Total Objects", sum(counter.values()))

        if counter:
            class_box.text("\n".join([f"{k}: {v}" for k, v in counter.items()]))
        else:
            class_box.text("No detections")

        # -------- CLEAN GPU --------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.01)

    cap.release()

