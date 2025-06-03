using System.Diagnostics;
using NumberplateRecognition.Entities;
using NumberplateRecognition.Services;
using OpenCvSharp;
using System.Text.Json;
using System.Threading.Channels;


// Sources
//var json = File.ReadAllText("E:\\Projects\\NumberplateRecognition\\NumberplateRecognition\\images.json");
//var paths = JsonSerializer.Deserialize<List<string>>(json)!;


var json = File.ReadAllText("camera.json");
var urls = JsonSerializer.Deserialize<List<string>>(json);

string detectionOnnxPath = Path.Combine("Models", "yolo11n.onnx");
const string detectionModelPath = "http://127.0.0.1:800#/detect";
const string recognitionModelPath = "http://127.0.0.1:16000/read";


List<Func<Task>> taskFactories = [];
List<Task> tasks = [];


Channel<Record> sharedChannel = Channel.CreateUnbounded<Record>();

ILicensePlateReader reader = new DtrReader(recognitionModelPath);
taskFactories.Add(
    () => Task.Run(async () =>
    {
        try
        {
            while (true)
            {
                using var record = await sharedChannel.Reader.ReadAsync();

                var timer = Stopwatch.StartNew();
                var result = await reader.ReadPlate(record.Frame);
                timer.Stop();
                Console.WriteLine($"Recognition latency: {timer.ElapsedMilliseconds}\n");

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
            Console.WriteLine($"{ex.Message}\n");
        }
    })
);


for (int x = 0; x < (urls?.Count?? 0); x++)
{
    if (String.IsNullOrEmpty(urls![x])) continue;

    int id = x;
    var channel = Channel.CreateBounded<Record>(new BoundedChannelOptions(capacity: 14) { FullMode = BoundedChannelFullMode.DropOldest });

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
                    var record = new Record(frame, id + 1);
                    await channel.Writer.WriteAsync(record);
                    await Task.Delay(4000);
                }*/


                var capture = new VideoCapture(urls[id]);
                if (!capture.IsOpened())
                {
                    Console.WriteLine($"Connection failed: {urls[id]}\n");
                }

                while (true)
                {
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
                        Console.WriteLine($"Reading stream failed or frame is empty: {urls[id]}\n");
                    }
                }
            }

            catch (Exception ex)
            {
                Console.WriteLine($"{ex.Message}\n");
            }
        })
    );



    // var model = new TorchModel(detectionModelPath.Replace("#", x.ToString()));
    var model = new OnnxModel(detectionOnnxPath);

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
                    Console.WriteLine($"Detection Latency: {timer.ElapsedMilliseconds}\n");
                    
                    if (result)
                    {
                        await sharedChannel.Writer.WriteAsync(record);
                    }
                    else record.Dispose();
                }
            }

            catch (Exception ex)
            {
                Console.WriteLine($"{ex.Message}\n");
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
                    Console.WriteLine($"Task restarted: {i}");
                    tasks[i] = taskFactories[i]();
                }
            }

            await Task.Delay(10000);
        }
    }

    catch (Exception ex)
    {
        Console.WriteLine(ex.Message);
    }
});



tasks.Add(superVisor);
Task.WhenAll(tasks).GetAwaiter().GetResult();
