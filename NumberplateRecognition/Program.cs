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

var config = JsonSerializer.Deserialize<AppConfig>(file) ?? throw new InvalidOperationException("Invalid configuration file.");

var logSetting = new Dictionary<string, string?>()
{
    ["Serilog:MinimumLevel:Default"] =  config.MinimumLogLevel,
    ["Serilog:WriteTo:0:Name"] = "Console",
    ["Serilog:WriteTo:1:Name"] = "Seq",
    ["Serilog:WriteTo:1:Args:serverUrl"] = config.Seq
};
IConfiguration serilog = new ConfigurationBuilder().AddInMemoryCollection(logSetting).Build();
Log.Logger = new LoggerConfiguration().ReadFrom.Configuration(serilog).CreateLogger();


// Lists

var cameras = config.Cameras;
cameras.RemoveAll(a => !a.IsActive);

List<VideoCapture> captures = [];
List<Object> locks = [];
List<Func<Task>> taskFactories = [];
List<Task> tasks = [];


// Reader Task

Channel<Record> sharedChannel = Channel.CreateUnbounded<Record>();

ILicensePlateReader reader = new DtrReader(config.RecognitionModelPath);
taskFactories.Add(
    () => Task.Run(async () =>
    {
        int i = 0;

        try
        {
            var notifyService = new NotifyService(config.ApiPath);

            while (true)
            {
                using var record = await sharedChannel.Reader.ReadAsync();
                var now = DateTime.Now;
                i = record.CameraIndex;
                Log.Debug("Got frame from camera #{id}", cameras[i].Id);

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
                    Cv2.ImWrite(Path.Combine(path, $"{now.Hour}-{now.Minute}-{now.Second}_{cameras[record.CameraIndex].Id}.jpg"), record.Frame);
                }

                else if (result.Item1 == "ClientError")
                {
                    string path = Path.Combine("log", "client_error", $"{now.Year}-{now.Month}-{now.Day}");
                    if (!Directory.Exists(path))
                    {
                        Directory.CreateDirectory(path);
                    }
                    Cv2.ImWrite(Path.Combine(path, $"{now.Hour}-{now.Minute}-{now.Second}_{cameras[record.CameraIndex].Id}.jpg"), record.Frame);
                }

                else if (result.Item2 >= 0.1 || result.Item1.Length == 8)
                {
                    var success = await notifyService.NotifyApi(Convert.ToInt32(cameras[i].Id), cameras[i].Address, cameras[i].Name, now, result.Item1, record.Frame, result.Item2);
                    if (!success)
                    {
                        string path = Path.Combine("log", "api_error", $"{now.Year}-{now.Month}-{now.Day}");
                        if (!Directory.Exists(path))
                        {
                            Directory.CreateDirectory(path);
                        }
                        Cv2.ImWrite(Path.Combine(path, $"{now.Hour}-{now.Minute}-{now.Second}_{cameras[record.CameraIndex].Id}.jpg"), record.Frame);
                    }
                }
            }
        }

        catch (Exception ex)
        {
            Log.Error(ex, "Recognition Error in camera #{id}", cameras[i].Id);
        }
    })
);



// Captures' initialization

for (int i = 0; i < cameras.Count; i++)
{
    if (!String.IsNullOrEmpty(cameras[i].Url))
    {
        captures.Add(new VideoCapture(cameras[i].Url));
        while (true)
        {
            if (captures[i].IsOpened()) break;
            captures[i] = new VideoCapture(cameras[i].Url);
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
            for (int i = 0; i < cameras.Count; i++)
            {
                lock (locks[i])
                {
                    bool success = captures[i].Grab();
                    if (!success)
                    {
                        Log.Error("Reading stream failed or frame is empty in camera: {val}   |   reconnecting to camera...", cameras[i].Url);
                        captures[i].Release();
                        captures[i].Dispose();
                        captures[i] = new VideoCapture(cameras[i].Url);
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

for (int i = 0; i < cameras.Count; i++)
{
    if (String.IsNullOrEmpty(cameras[i].Url)) continue;

    int ii = i;
    var channel = Channel.CreateBounded<Record>(new BoundedChannelOptions(capacity: 14) { FullMode = BoundedChannelFullMode.DropOldest });

    ITruckDetectorModel model;
    if (config.DetectionModelType == "ot") model = new OnnxTruckDetectionModel(config.DetectionModelPath);
    else if (config.DetectionModelType == "tt") model = new TorchTruckDetectionModel(config.DetectionModelPath.Replace("#", ii.ToString("D3")));
    else model = new TorchPlateDetectionModel(config.DetectionModelPath.Replace("#", ii.ToString("D3")));

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
                Log.Error(ex, "Detection Task Error in camera #{id} with model {model} in camera: {val}", cameras[ii].Id, config.DetectionModelPath, cameras[ii].Url);
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
                    lock (locks[ii])
                    {
                        success = captures[ii].Retrieve(frame);
                    }

                    if (success && !frame.Empty())
                    {
                        var record = new Record(frame, ii);
                        await channel.Writer.WriteAsync(record);
                        await Task.Delay(1000);
                    }

                    else
                    {
                        Log.Error("Reading stream failed or frame is empty in camera: {val}   |   reconnecting to camera...", cameras[ii].Url);
                        frame.Dispose();

                        lock (locks[ii])
                        {
                            captures[ii].Release();
                            captures[ii].Dispose();
                            captures[ii] = new VideoCapture(cameras[ii].Url);
                        }
                        await Task.Delay(2000);
                    }
                }
            }

            catch (Exception ex)
            {
                Log.Error(ex, "StreamReader Task Error in camera #{id}", cameras[ii].Id);
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
