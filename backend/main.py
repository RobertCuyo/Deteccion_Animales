from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import os

app = FastAPI(title="Detector de Animales - YOLOv8")

model = YOLO("yolov8n.pt")

animal_classes = {
    "dog", "cat", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

INPUT_DIR = "backend/videos/input"
OUTPUT_DIR = "backend/videos/output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================
# ENDPOINT RAÍZ 
# =========================
@app.get("/")
def root():
    return {"status": "Backend YOLO activo"}

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):

    # =========================
    # GUARDAR VIDEO ORIGINAL
    # =========================
    video_id = str(uuid.uuid4())
    input_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_detected.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # =========================
    # ABRIR VIDEO
    # =========================
    cap = cv2.VideoCapture(input_path)
    

    if not cap.isOpened():
        return {"error": "No se pudo abrir el video"}
    
    detections_count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # =========================
    # PROCESAR FRAME A FRAME
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, iou=0.45, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]

                if name in animal_classes:
                    
                    detections_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"{name} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        out.write(frame)

    cap.release()
    out.release()

    # =========================
    # DEVOLVER VIDEO PROCESADO
    # =========================
    if detections_count > 0:
        final_name  = f"{video_id}_HAS_ANIMAL.mp4"
    else:
        final_name  = f"{video_id}_NO_ANIMAL.mp4"

    final_output_path = os.path.join(OUTPUT_DIR, final_name)
    os.rename(output_path, final_output_path)

    return FileResponse(
        final_output_path,
        media_type="video/mp4",
        filename=final_name 
    )
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"detections": []}

    results = model(img, conf=0.4, verbose=False)

    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            if name in animal_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append({
                    "animal": name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

    return {"detections": detections}

    