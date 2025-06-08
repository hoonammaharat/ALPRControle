import sys
import random

import fastapi
import uvicorn

import numpy as np
import cv2
import torch
import ultralytics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
license_plate_detector = ultralytics.YOLO("./models/license_plate_detector.pt")
license_plate_detector.to(device)
print(f"using device: {device}")

app = fastapi.FastAPI()


x = 0

@app.post("/detect")
async def detect(request: fastapi.Request):
    shape = tuple(map(int, request.headers.get("shape").split(",")))

    raw_bytes = await request.body()
    image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)

    license_plates = license_plate_detector.predict(image)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        if score > 0.5:
            return { "Result": "true"}

    return { "Result": "false"}


if __name__ == "__main__":
    port = int(sys.argv[1])
    uvicorn.run(app, host="127.0.0.1", port=(8000 + port))
