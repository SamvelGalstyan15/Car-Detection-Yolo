from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import PIL
import io

app = FastAPI(title="YOLOv8 Car Detection API")

model = YOLO("best.pt")

@app.post("/predict")
async def predict_car(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = PIL.open(io.BytesIO(image_bytes))
    
    results = model(image)
    
    detections = []
    for i in results:
        for j in i.boxes:
            detections.append({
                "class": int(j.cls),
                "name": model.names[int(j.cls)],
                "confidence": round(float(j.conf), 2),
                "bbox": [round(float(x), 1) for x in j.xyxy[0]] 
            })
            
    return {"detections": detections}
