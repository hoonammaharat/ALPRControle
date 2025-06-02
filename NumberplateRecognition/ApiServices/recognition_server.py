import argparse

import fastapi
import uvicorn

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from deep_text_recognition_benchmark.dtrb import DTRB

parser = argparse.ArgumentParser()
# parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
# parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--detector-weights', type=str, default=r".\models\license_plate_detector.pt")
parser.add_argument('--recognizer-weights', type=str, default=r".\models\dtrb-None-VGG-BiLSTM-CTC-license-plate-recognizer.pth")
parser.add_argument('--input-image', type=str, default=r".\example.jpg")
parser.add_argument('--threshold', type=float, default=0.5)

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plate_recognizer = DTRB("./models/dtrb-None-VGG-BiLSTM-CTC-license-plate-recognizer.pth", opt)

license_plate_detector = YOLO("./models/license_plate_detector.pt")
license_plate_detector.to(device)

# plate_detector = YOLO(opt.detector_weights)
# plate_detector.to(device)

opt.imgH = 30
opt.imgW = 80
opt.PAD = True


app = fastapi.FastAPI()


@app.post("/read")
async def read(request: fastapi.Request):
    shape = tuple(map(int, request.headers.get("shape").split(",")))

    raw_bytes = await request.body()
    image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)

    license_plates = license_plate_detector.predict(image)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        if score > opt.threshold:
            plate_crop = image[int(y1):int(y2), int(x1):int(x2), :]
            plate_crop = cv2.resize(plate_crop, (100, 32))
            plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            text, confidence = plate_recognizer.predict(plate_crop, opt)
            if False:
                return { "Result": "Unreadable", "Confidence": 0.0 }
            return { "Result": text, "Confidence": float(confidence) }

    # result = plate_detector.predict(image)[0]
    # print("plates:        ", len(result))
    # print(result.boxes.data)
    # for i in range(len(result.boxes.xyxy)):
    #     if result.boxes.conf[i] > opt.threshold:
    #         bbox = result.boxes.xyxy[i]
    #         bbox = bbox.cpu().detach().numpy().astype(int)
    #         x1, y1, x2, y2 = bbox
    #         plate_image = image[y1:y2, x1:x2].copy()
    #         plate_image = cv2.resize(plate_image, (100, 32))
    #         plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    #
    #         text = plate_recognizer.predict(plate_image, opt)
    #         return { "Result": text, "Confidence": 0.0}

    return { "Result": "NotFound", "Confidence": 0.0 }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=16000)
