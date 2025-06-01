using System.Diagnostics;
using NumberplateRecognition.Entities;
using NumberplateRecognition.Services;
using OpenCvSharp;
using System.Text.Json;
using System.Threading.Channels;


// Sources

var json = File.ReadAllText("E:\\Projects\\NumberplateRecognition\\NumberplateRecognition\\images.json");
var paths = JsonSerializer.Deserialize<List<string>>(json)!;


List<string> insideCamUrls = [];
List<string> outsideCamUrls = [];


const string detectionOnnxPath = "E:\\Projects\\NumberplateRecognition\\NumberplateRecognition\\Models\\yolo11n.onnx";
const string detectionModelPath = "http://127.0.0.1:800#/detect";
const string recognitionModelPath = "http://127.0.0.1:16000/read";


List<Func<Task>> taskFactories = [];
List<Task> tasks = [];



// Actual Algoritm for running app properly and concurrently:


Channel<Record> sharedChannel = Channel.CreateUnbounded<Record>();

ILicensePlateReader reader = new DtrReader(recognitionModelPath);
taskFactories.Add(
    () => Task.Run(async () =>
    {
        try
        {
            while (true)
            {
                var record = await sharedChannel.Reader.ReadAsync();
                var result = await reader.ReadPlate(record.Frame);
                if (result == "None")
                {

                }

                record.Dispose();
            }
        }

        catch (Exception ex)
        {
            Console.WriteLine(ex);
        }
    })
);


for (int x = 0; x < 1; x++)
{
    int id = x;
    var channel = Channel.CreateBounded<Record>(new BoundedChannelOptions(capacity: 14) { FullMode = BoundedChannelFullMode.DropOldest });

    taskFactories.Add(
        () => Task.Run(async () =>
        {
            try
            {
                // Debugging model execution in concurrent mode without network streaming execution:

                foreach (var path in paths)
                {
                    Console.WriteLine("File: " + path + "\n");
                    var frame = Cv2.ImRead(path);
                    var record = new Record(frame, id + 1);
                    await channel.Writer.WriteAsync(record);
                    await Task.Delay(4000);
                }


                // Real algorithm with streaming and all features:

                /*var capture = new VideoCapture(insideCamURLs[x]);
                if (!capture.IsOpened())
                {
                    Console.WriteLine("Connection failed!");
                }

                while (true)
                {
                    var frame = new Mat();
                    bool success = capture.Read(frame);
                    if (success && !frame.Empty())
                    {
                        var record = new Record(frame, id);
                        await channel.Writer.WriteAsync(record);
                        await Task.Delay(450);
                    }
                    else
                    {
                        Console.WriteLine("Reading stream failed or frame is empty!");
                    }
                }*/
            }

            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        })
    );



    taskFactories.Add(
        () => Task.Run(async () =>
        {
            try
            {
                // Debugging model execution in concurrent mode without network streaming execution:

                foreach (var path in paths)
                {
                    Console.WriteLine("File: " + path + "\n");
                    var frame = Cv2.ImRead(path);
                    var record = new Record(frame, -id - 1);
                    await channel.Writer.WriteAsync(record);
                    await Task.Delay(4000);
                }


                // Real algorithm with streaming and all features:

                /*var capture = new VideoCapture(outsideCamURLs[x]);
                if (!capture.IsOpened())
                {
                    Console.WriteLine("Connection failed!");
                }

                while (true)
                {
                    var frame = new Mat();
                    bool success = capture.Read(frame);
                    if (success && !frame.Empty())
                    {
                        var record = new Record(frame, -id);
                        await channel.Writer.WriteAsync(record);
                        await Task.Delay(450);
                    }
                    else
                    {
                        Console.WriteLine("Reading stream failed or frame is empty!");
                    }
                }*/
            }

            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        })
    );



    ITruckDetectorModel model;  // new TorchModel(detectionModelPath.Replace("#", x.ToString()));
    model = new OnnxModel(detectionOnnxPath);

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
                    Console.WriteLine(timer.ElapsedMilliseconds);

                    if (result)
                    {
                        // await sharedChannel.Writer.WriteAsync(record);
                    }
                    else
                    {
                        record.Dispose();
                    }
                }
            }

            catch (Exception ex)
            {
                Console.WriteLine(ex);
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
                    Console.WriteLine($"Task Restarted: {i}");
                    tasks[i] = taskFactories[i]();
                }
            }

            await Task.Delay(10000);
        }
    }

    catch (Exception ex)
    {
        Console.WriteLine(ex);
    }
});



tasks.Add(superVisor);
Task.WhenAll(tasks).GetAwaiter().GetResult();
