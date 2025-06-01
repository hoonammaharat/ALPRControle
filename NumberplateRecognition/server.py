import fastapi
import uvicorn
import torch
import ultralytics
from ultralytics.utils.ops import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ultralytics.YOLO("./Models/yolo11n.pt")
model.to(device)
print(f"using device: {device}")

app = fastapi.FastAPI()

@app.post("/detect")
async def detect(request: fastapi.Request):
    body = await request.json()
    tensor = torch.tensor(body["image"], dtype=torch.float32, device=device)
    with torch.no_grad():
        raw_output = model.model(tensor)
        output = non_max_suppression(raw_output)
    output = [output[0].cpu().tolist()]
    return { "Shape": [len(output), len(output[0]), len(output[0][0])], "Output": output }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
