from ultralytics import YOLO
import cv2
import time

# ==========================
# CONFIGURACIÓN
# ==========================

model = YOLO("yolov8n.pt")

animal_classes = {
    "dog", "cat", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

# ==========================
# FUENTE DE VIDEO (CELULAR)
# ==========================
STREAM_URL = "http://192.168.100.3:8080/video"
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara del celular")
    exit()

print("✅ Cámara del celular conectada")

# ==========================
# PROPIEDADES DEL STREAM
# ==========================

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPS fijo (recomendado para streams IP)
OUTPUT_FPS = 20

print(f"📐 Resolución stream: {WIDTH}x{HEIGHT}")
print(f"🎞️ FPS salida: {OUTPUT_FPS}")

# ==========================
# VIDEO DE SALIDA
# ==========================

output_path = "videos/celular_detectado.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    OUTPUT_FPS,
    (WIDTH, HEIGHT)
)

print(f"💾 Guardando video en: {output_path}")

# ==========================
# LOOP PRINCIPAL
# ==========================

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame no recibido (stream cortado)")
        break

    # Inferencia YOLO
    results = model(frame, conf=0.4, verbose=False)

    animal_detectado = False

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            if name in animal_classes:
                animal_detectado = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # FPS visual
    curr_time = time.time()
    fps_vis = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps_vis)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    if animal_detectado:
        cv2.putText(
            frame,
            "ANIMAL DETECTADO",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Detección de Animales - Cámara Celular", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# LIMPIEZA
# ==========================

cap.release()
out.release()
cv2.destroyAllWindows()

print("🎉 Video del celular guardado correctamente")