from ultralytics import YOLO

# Load model (YOLOv8 format)
model = YOLO('../Lego/bestTrained.pt')  # Replace with your trained model if you have one

def detect_objects(frame):
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                'class': cls,
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            })

    return detections
