import sys

import fastapi
import uvicorn

import torch
import ultralytics
from ultralytics.utils.ops import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ultralytics.YOLO("./models/yolo11n.pt")
model.to(device)
print(f"using device: {device}")

app = fastapi.FastAPI()


@app.post("/detect")
async def detect(request: fastapi.Request):
    shape = tuple(map(int, request.headers.get("shape").split(",")))

    raw_bytes = await request.body()
    bytes = torch.frombuffer(raw_bytes, dtype=torch.uint8)

    tensor = bytes.reshape(shape[0], shape[1], shape[2], shape[3]).to(dtype=torch.float32, device=device).div_(255.0)
    with torch.no_grad():
        raw_output = model.model(tensor)
        output = non_max_suppression(raw_output)


    for b in range(0, output[0].shape[0]):
        if output[0][b][4] > 0.3 and output[0][b][5] == 7:
            return { "Result": "true" }

    return { "Result": "false"}


if __name__ == "__main__":
    port = int(sys.argv[1])
    uvicorn.run(app, host="127.0.0.1", port=(8000 + port))
