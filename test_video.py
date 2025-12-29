import cv2

cap = cv2.VideoCapture("videos/perro.mp4")

if not cap.isOpened():
    print("❌ OpenCV no puede abrir el video")
else:
    print("✅ Video cargado correctamente")

cap.release()