import streamlit as st
import cv2
import time
import tempfile
import psutil

from utils.model_factory import ModelFactory

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Detection Dashboard",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("AI Model Inference Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=["pt"])
video_option = st.sidebar.selectbox("Video Source", ["Webcam", "Upload Video"])

video_file = None
if video_option == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

start = st.sidebar.button("▶ Start Detection")

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([3, 1])

frame_window = col1.empty()

col2.subheader("📊 Performance")
latency_box = col2.empty()
fps_box = col2.empty()
cpu_box = col2.empty()

# ---------------- RUN ----------------
if start:

    if not model_file:
        st.error("Please upload a model file")
        st.stop()

    # -------- SAVE MODEL --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    # -------- LOAD MODEL --------
    handler = ModelFactory.create(model_path, device="cpu")

    # -------- VIDEO SOURCE --------
    if video_option == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if not video_file:
            st.error("Please upload a video file")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = tmp_vid.name

        cap = cv2.VideoCapture(video_path)

    prev_time = time.time()

    # ---------------- LOOP ----------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # -------- PREDICTION --------
        results = handler.predict(frame)

        # -------- DRAW --------
        for det in results:
            for box in det["boxes"]:
                x1, y1, x2, y2 = box["bbox"]
                label = f"{box['class_name']} {box['conf']:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box["color"], 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            box["color"], 2)

        # -------- METRICS --------
        latency = (time.time() - start_time) * 1000
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cpu = psutil.cpu_percent()

        # -------- DISPLAY --------
        frame_window.image(frame, channels="BGR", use_container_width=True)

        latency_box.metric("Latency (ms)", f"{latency:.2f}")
        fps_box.metric("FPS", f"{fps:.2f}")
        cpu_box.metric("CPU (%)", f"{cpu}")

    cap.release()
