import requests

url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/videos/dog.mp4"
ruta = "videos/prueba.mp4"

response = requests.get(url)
with open(ruta, "wb") as f:
    f.write(response.content)

print("✅ Video descargado en", ruta)