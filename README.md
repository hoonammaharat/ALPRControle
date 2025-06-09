It's a console application which can connect to cc cameras,
detect licence plates and read Persian number plates.
This app uses OpenCvSharp for connecting to cameras through RTSP protocol.
Then it uses a YOLO v8n fine tuned model for detecting plates on the picture,
and finally uses a deep text recognition model specially trained for Persian numbers recognition for reading text.

This app has a modular architecture, and designed for stability and fault tolerance.
App consists of five parts:

First it starts with loading configurations for cameras, models, APIs, etc.

Then it runs a reader task parallelly, this task includes a channel for receiving frames which are confirmed to have a plate within,
a model service object for reading existing plates in frames, and a notifier service object to inform API service about car passing.

Then it runs a frame advance task, this task create connection object for each url in settings and grabs frames in buffer to keep frame fresh always.
It add all created connection to a list called captures.

Then it loop over urls; for each url a task is started with a model service object to detect plates and a channel to receive last frames.
Also it runs another task, it retrieve last frames from related camera in captures list, and write it to channel.
This task has a manual delay for keep distance between frames, normally 0.5 or 1 second.
This task and frame advance task both check connection repeatedly and reconnect if it fails.

All above explained tasks are wrapped in another lambda function called task factory and added to a list.
Task factories are used to restarting a task with the same captured variables if task has stopped for any reason.
First a loop calls all lambda functions to run tasks, and add their handle to another list.
In the end of program another task called supervisor starts and check all tasks in tasks list.
If a task is not in running status, it calls corresponding element in task Factories list to restart failed task, and put its handle to the same index in tasks list.
This process repeats every 10 second.

About models; for initial detection we can choose YOLO model as an onnx model or a pytorch model in an external python API either.
ONNX model is embedded in C# and has lower latency for cpu, but on gpu onnx runtime is not fully compatible with cuda, so it's slower.
Using pytorch model makes inference operation about 10X faster, so we use it through localhost with a octet-stream content type for efficiency.
We can choose a truck detection service and plate detection service, truck detection service enables us to filter trucks,
but it's not that accurate if whole truck body is not in the image frame. So using plate detection service directly is safer choice.

For text recognition purpose we have another model designed with DeepTextRecognition framework tuned for Persian plates.
It's available via a python API service too.

Python API services are built with FastAPI framework, which is the fastest. Truck/Plate detection service runs on localhost:8000 + arg/detect ;
for starting service you should give it arg at shell command. This number is added to port of server address.
Considering each camera has its own model, you may need run several service instances for supporting them all without latency.
Notice that dotnet app expect services to be running on port 8000 -> 8000 + n - 1; n is number of camera urls.
Text recognition service runs on localhost:16000/read ; it's singleton and shared between all cameras due to non-permanent inputs and for less resource consumption, specially ram. 

For fault tolerance, all execution blocks in app are wrapped in try catch block, it let's tasks fail never.
All errors are logged with serilog in a seq server.
After reading plate text it gets evaluated and sent to server with all data including self image, but if api call fails the image will be logged in log/api_error/Date/Time_CamId.jpg. 
Other unpredicted errors will be logged in the same directory with different subfolder.

All apps and services are written concurrent using async await facilities relying on CLR scheduler.
It's necessary to keep latency lowest and efficiency most.
