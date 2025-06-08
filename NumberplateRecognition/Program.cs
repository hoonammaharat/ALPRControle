using System.Diagnostics;
using System.Text.Json;
using System.Threading.Channels;

using Microsoft.Extensions.Configuration;
using OpenCvSharp;
using Serilog;

using NumberplateRecognition.Entities;
using NumberplateRecognition.Services;


// Configurations

var file = File.ReadAllText("config.json");
var config = JsonSerializer.Deserialize<Dictionary<string, string>>(file);

var urls = config!["CameraUrls"].Split("|");
var detectionModelType = config["DetectionModelType"];
var detectionModelPath = config["DetectionModelPath"];
var recognitionModelPath = config["RecognitionModelPath"];
var apiPath = config["ApiPath"];

var logSetting = new Dictionary<string, string?>()
{
    ["Serilog:MinimumLevel:Default"] =  config["MinimumLogLevel"],
    ["Serilog:WriteTo:0:Name"] = "Seq",
    ["Serilog:WriteTo:0:Args:serverUrl"] = config["Seq"]
};
IConfiguration serilog = new ConfigurationBuilder().AddInMemoryCollection(logSetting).Build();
Log.Logger = new LoggerConfiguration().ReadFrom.Configuration(serilog).CreateLogger();


// Lists

List<VideoCapture> captures = [];
List<Object> locks = [];
List<Func<Task>> taskFactories = [];
List<Task> tasks = [];



// Reader Task

Channel<Record> sharedChannel = Channel.CreateUnbounded<Record>();

ILicensePlateReader reader = new DtrReader(recognitionModelPath);
taskFactories.Add(
    () => Task.Run(async () =>
    {
        int id = 0;

        try
        {
            var notifyService = new NotifyService(apiPath);

            while (true)
            {
                var now = DateTime.Now;

                using var record = await sharedChannel.Reader.ReadAsync();
                id = record.CameraID;
                Log.Debug("Got frame from camera #{id}: {val}", id, urls![id]);

                var timer = Stopwatch.StartNew();
                (string, float) result = await reader.ReadPlate(record.Frame);
                timer.Stop();
                Log.Debug("Result: {r}  |  Recognition latency: {t}", $"{result.Item1}  |  {result.Item2}", timer.ElapsedMilliseconds);

                if (result.Item1 == "ServiceError" || result.Item1 == "InternalServiceError")
                {
                    string path = Path.Combine("log", "service_error", $"{now.Year}-{now.Month}-{now.Day}");
                    if (!Directory.Exists(path))
                    {
                        Directory.CreateDirectory(path);
                    }
                    Cv2.ImWrite(Path.Combine(path, $"{now.Hour}-{now.Minute}-{now.Second}_{record.CameraID}.jpg"), record.Frame);
                }

                else if (result.Item1 == "ClientError")
                {
                    string path = Path.Combine("log", "client_error", $"{now.Year}-{now.Month}-{now.Day}");
                    if (!Directory.Exists(path))
                    {
                        Directory.CreateDirectory(path);
                    }
                    Cv2.ImWrite(Path.Combine(path, $"{now.Hour}-{now.Minute}-{now.Second}_{record.CameraID}.jpg"), record.Frame);
                }

                else
                {
                    var success = await notifyService.NotifyApi(record.CameraID, urls[id], result.Item1, record.Frame, result.Item2);
                    if (!success)
                    {
                        string path = Path.Combine("log", "api_error", $"{now.Year}-{now.Month}-{now.Day}");
                        if (!Directory.Exists(path))
                        {
                            Directory.CreateDirectory(path);
                        }
                        Cv2.ImWrite(Path.Combine(path, $"{now.Hour}-{now.Minute}-{now.Second}_{record.CameraID}.jpg"), record.Frame);
                    }
                }
            }
        }

        catch (Exception ex)
        {
            Log.Error(ex, "Recognition Error in camera #{id}:  {val}", id, urls![id]);
        }
    })
);



// Captures' initialization

for (int x = 0; x < (urls?.Length ?? 0); x++)
{
    if (!String.IsNullOrEmpty(urls![x]))
    {
        captures.Add(new VideoCapture(urls![x]));
        while (true)
        {
            if (captures[x].IsOpened()) break;
            captures[x] = new VideoCapture(urls![x]);
            Thread.Sleep(500);
        }

        locks.Add(new Object());
    }
}


// Frame Advance Task

taskFactories.Add(() => Task.Run(async () =>
{
    try
    {
        while (true)
        {
            for (int x = 0; x < (urls?.Length ?? 0); x++)
            {
                lock (locks[x])
                {
                    bool success = captures[x].Grab();
                    if (!success)
                    {
                        Log.Error("Reading stream failed or frame is empty in camera: {val}   |   reconnecting to camera...", urls![x]);
                        captures[x].Release();
                        captures[x].Dispose();
                        captures[x] = new VideoCapture(urls[x]);
                    }
                }
            }
            await Task.Delay(20); 
        }
    }

    catch (Exception ex)
    {
        Log.Error(ex, "Frame Advance Error occured.");
    }
}));



// Camera Tasks

for (int x = 0; x < (urls?.Length ?? 0); x++)
{
    if (String.IsNullOrEmpty(urls![x])) continue;

    int id = x;
    var channel = Channel.CreateBounded<Record>(new BoundedChannelOptions(capacity: 14) { FullMode = BoundedChannelFullMode.DropOldest });

    ITruckDetectorModel model;
    if (detectionModelType == "ot") model = new OnnxTruckDetectionModel(detectionModelPath);
    else if (detectionModelType == "tt") model = new TorchTruckDetectionModel(detectionModelPath.Replace("#", id.ToString()));
    else model = new TorchPlateDetectionModel(detectionModelPath.Replace("#", id.ToString()));

    taskFactories.Add(
        () => Task.Run(async () =>
        {
            Record? record = null;

            try
            {
                while (true)
                {
                    record = await channel.Reader.ReadAsync();

                    var timer = Stopwatch.StartNew();
                    var result = await model.DetectTruck(record.Frame);
                    timer.Stop();
                    Log.Debug("Truck Motion: {b}  |  Detection Latency: {t}", result, timer.ElapsedMilliseconds);

                    if (result)
                    {
                        await sharedChannel.Writer.WriteAsync(record);
                        await Task.Delay(5000);
                        while (channel.Reader.TryRead(out _));
                    }
                    else record.Dispose();
                }
            }

            catch (Exception ex)
            {
                Log.Error(ex, "Detection Task Error in camera #{id} with model {model} in camera: {val}", id, detectionModelPath, urls[id]);
                record?.Dispose();
            }
        })
    );


    
    taskFactories.Add(
        () => Task.Run(async () =>
        {
            Mat? frame = null;
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

                
                while (true)
                {
                    frame = new Mat();
                    bool success;
                    lock (locks[id])
                    {
                        success = captures[id].Retrieve(frame);
                    }

                    if (success && !frame.Empty())
                    {
                        var record = new Record(frame, id);
                        await channel.Writer.WriteAsync(record);
                        await Task.Delay(1000);
                    }

                    else
                    {
                        Log.Error("Reading stream failed or frame is empty in camera: {val}   |   reconnecting to camera...", urls[id]);
                        frame.Dispose();

                        lock (locks[id])
                        {
                            captures[id].Release();
                            captures[id].Dispose();
                            captures[id] = new VideoCapture(urls[id]);
                        }
                        await Task.Delay(2000);
                    }
                }
            }

            catch (Exception ex)
            {
                Log.Error(ex, "StreamReader Task Error in camera #{id}: {val}", id, urls[id]);
                frame?.Dispose();
            }
        })
    );
}



// Running Tasks

foreach (var factory in taskFactories)
{
    tasks.Add(factory());
    Thread.Sleep(250);
}



// Supervisor Task

var supervisor = Task.Run(async () =>
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
        Log.Error(ex, "Supervisor Error occured.");
    }
});



tasks.Add(supervisor);
Task.WhenAll(tasks).GetAwaiter().GetResult();
