from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Detector de Animales - YOLOv8")

# Cargar modelo una sola vez
model = YOLO("yolov8n.pt")

animal_classes = {
    "dog", "cat", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

@app.get("/")
def root():
    return {"status": "Backend YOLO activo"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    # Leer imagen
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

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

    return {
        "num_detections": len(detections),
        "detections": detections
    }