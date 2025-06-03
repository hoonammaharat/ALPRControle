using System.Diagnostics;
using System.Text.Json;
using System.Threading.Channels;

using Serilog;
using OpenCvSharp;

using NumberplateRecognition.Entities;
using NumberplateRecognition.Services;


// Sources

var json = File.ReadAllText("E:\\Projects\\NumberplateRecognition\\NumberplateRecognition\\images.json");
var paths = JsonSerializer.Deserialize<List<string>>(json)!;

var jsonUrls = File.ReadAllText("camera.json");
var urls = JsonSerializer.Deserialize<List<string>>(jsonUrls);


string detectionOnnxPath = Path.Combine("Models", "yolo11n.onnx");
const string detectionModelPath = "http://127.0.0.1:800#/detect";
const string recognitionModelPath = "http://127.0.0.1:16000/read";

bool onnx = true;
Console.Write("Enter 0 if you want to execute detection model on external python service using native cuda APIs via torch, else enter any other key: ");
var input = Console.ReadLine();

if (input == "0")
{
    onnx = false;
}


List<Func<Task>> taskFactories = [];
List<Task> tasks = [];


Log.Logger = new LoggerConfiguration().MinimumLevel.Debug().WriteTo.Seq("http://localhost:5341").Enrich.FromLogContext().CreateLogger();



Channel<Record> sharedChannel = Channel.CreateUnbounded<Record>();

ILicensePlateReader reader = new DtrReader(recognitionModelPath);
taskFactories.Add(
    () => Task.Run(async () =>
    {
        int id = 0;

        try
        {
            while (true)
            {
                using var record = await sharedChannel.Reader.ReadAsync();
                id = record.CameraID;
                Log.Debug("Got frame from camera #{id}: {val}", id, urls![id]);

                var timer = Stopwatch.StartNew();
                var result = await reader.ReadPlate(record.Frame);
                timer.Stop();
                Log.Debug("Result: {r}  |  Recognition latency: {t}", result, timer.ElapsedMilliseconds);

                if (result == "ServiceError" || result == "InternalServiceError")
                {
                    Console.WriteLine("Service doesn't respond correctly or is unavailable.\n");
                }

                else if (result == "ClientError")
                {
                    Console.WriteLine("An error occured in internal app's services, see logs.\n");
                }

                else if (result == "NotFound")
                {
                    Console.WriteLine($"License plate not found in detected frame on camera: {record.CameraID}\n");
                }

                else
                {
                    Console.WriteLine($"A license plate detected in camera: {record.CameraID}\nDetected text: {result}\n");
                }
            }
        }

        catch (Exception ex)
        {
            Log.Error(ex, "Recognition Error in camera #{id}:  {val}", id, urls![id]);
        }
    })
);



for (int x = 0; x < (urls?.Count?? 0); x++)
{
    if (String.IsNullOrEmpty(urls![x])) continue;

    int id = x;
    var channel = Channel.CreateBounded<Record>(new BoundedChannelOptions(capacity: 14) { FullMode = BoundedChannelFullMode.DropOldest });

    ITruckDetectorModel model;
    if (onnx) model = new OnnxModel(detectionOnnxPath);
    else model = new TorchModel(detectionModelPath.Replace("#", id.ToString()));


    taskFactories.Add(
        () => Task.Run(async () =>
        {
            try
            {
                while (true)
                {
                    var record = await channel.Reader.ReadAsync();

                    var timer = Stopwatch.StartNew();
                    var result = await model.DetectTruck(record.Frame);
                    timer.Stop();
                    Log.Debug("Truck Motion: {b}  |  Detection Latency: {t}", result, timer.ElapsedMilliseconds);
                    
                    if (result)
                    {
                        await sharedChannel.Writer.WriteAsync(record);
                    }
                    else record.Dispose();
                }
            }

            catch (Exception ex)
            {
                Log.Error(ex, "Detection Task Error in camera #{id} with model {model} in camera: {val}", id, (onnx)? detectionOnnxPath : detectionModelPath, urls[id]);
            }
        })
    );


    
    taskFactories.Add(
        () => Task.Run(async () =>
        {
            try
            {
                // Debugging model execution in concurrent mode without network streaming execution:

                /*foreach (var path in paths)
                {
                    Console.WriteLine("File: " + path + "\n");
                    var frame = Cv2.ImRead(path);
                    var record = new Record(frame, id);
                    await channel.Writer.WriteAsync(record);
                    await Task.Delay(500);
                }*/


                var capture = new VideoCapture(urls[id]);

                while (true)
                {
                    if (!capture.IsOpened())
                    {
                        Log.Error("Connection failed to camera: {val}", urls[id]);
                        await Task.Delay(5000);
                        capture = new VideoCapture(urls[id]);
                        continue;
                    }

                    var frame = new Mat();
                    bool success = capture.Read(frame);
                    if (success && !frame.Empty())
                    {
                        var record = new Record(frame, id);
                        await channel.Writer.WriteAsync(record);
                        await Task.Delay(500);
                    }
                    else
                    {
                        Log.Error("Reading stream failed or frame is empty in camera: {val}", urls[id]);
                        await Task.Delay(1800);
                    }
                }
            }

            catch (Exception ex)
            {
                Log.Error(ex, "StreamReader Task Error in camera #{id}: {val}", id, urls[id]);
            }
        })
    );
}



foreach (var factory in taskFactories)
{
    tasks.Add(factory());
    Thread.Sleep(250);
}



var superVisor = Task.Run(async () =>
{
    try
    {
        while (true)
        {
            for (int i = 0; i < tasks.Count - 1; i++)
            {
                if (tasks[i].IsFaulted || tasks[i].IsCompleted)
                {
                    Log.Warning("Task restarted: {i}", i);
                    tasks[i] = taskFactories[i]();
                }
            }

            await Task.Delay(10000);
        }
    }

    catch (Exception ex)
    {
        Log.Error(ex, "Supervisor Error");
    }
});



tasks.Add(superVisor);
Task.WhenAll(tasks).GetAwaiter().GetResult();
