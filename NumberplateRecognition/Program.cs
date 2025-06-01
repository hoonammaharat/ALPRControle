using NumberplateRecognition.Entities;
using NumberplateRecognition.Services;
using OpenCvSharp;
using System.Text.Json;
using System.Threading.Channels;


// Sources

var json = File.ReadAllText("E:\\Projects\\NumberplateRecognition\\NumberplateRecognition\\images.json");
var pathes = JsonSerializer.Deserialize<List<string>>(json)!;

List<string> insideCamURLs = [];
List<string> outsideCamURLs = [];

string modelPath = "E:\\Projects\\NumberplateRecognition\\NumberplateRecognition\\Models\\yolo11n.onnx";

List<Task> tasks = [];
Channel<Record> sharedChannel = Channel.CreateUnbounded<Record>();


// Manual and direct path for debugging Model execution without concurrency complexities:

//string openVino = "OpenVINO";
//var OpenVINOOption = new Dictionary<string, string>() { { "enable_opencl_throttling", "true" }, { "device_type", "GPU" } };

//ITruckDetector model = new Model(modelPath, openVino, OpenVINOOption);

//foreach (var path in pathes)
//{
//    Console.WriteLine("File: " + path + "\n");
//    var frame = Cv2.ImRead(path);
//    Cv2.CvtColor(frame, frame, ColorConversionCodes.BGR2RGB);
//    model.DetectTruck(frame);
//}



// Actual Algoritm for running app properly and concurrently:

INumberplateReader reader = null;
Task NumberplateReaderThread = Task.Run(async () => { 
    while (true)
    {
        var record = await sharedChannel.Reader.ReadAsync();
        var result = reader.ReadNumberplate(record.Frame);

        record.Dispose();
    }
});


for (int x = 0; x < 4; x++)
{
    int id = x;
    var channel = Channel.CreateBounded<Record>(new BoundedChannelOptions(capacity: 14) { FullMode = BoundedChannelFullMode.DropOldest });

    var inCam = Task.Run(async () =>
    {
        // Debugging model execution in concurrent mode without network streaming execution:

        while (true) foreach(var path in pathes)
        {
            Console.WriteLine("File: " + path + "\n");
            var frame = Cv2.ImRead(path);
            var record = new Record(frame, id);
            await channel.Writer.WriteAsync(record);
            await Task.Delay(4000);
        }


        // Real algoritm with streaming and all features:

        var capture = new VideoCapture(insideCamURLs[x]);
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
        }
    });

    tasks.Add(inCam);

    Thread.Sleep(1000);  // to keep distance between cams



    var outCam = Task.Run(async () =>
    {
        // Debugging model execution in concurrent mode without network streaming execution:

        while (true) foreach (var path in pathes)
        {
                Console.WriteLine("File: " + path + "\n");
                var frame = Cv2.ImRead(path);
                var record = new Record(frame, -id);
                await channel.Writer.WriteAsync(record);
                await Task.Delay(4000);
        }


        // Real algoritm with streaming and all features:

        var capture = new VideoCapture(outsideCamURLs[x]);
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
        }
    });

    tasks.Add(outCam);



    ITruckDetectorModel model = new OnnxModel(modelPath);

    var detector = Task.Run(async () =>
    {
        while (true)
        {
            var record = await channel.Reader.ReadAsync();

            Console.WriteLine(record.CameraID);
            await model.DetectTruck(record.Frame);

            if (false)
            {
                await sharedChannel.Writer.WriteAsync(record);
            } else record.Dispose();
        }
    });

    tasks.Add(detector);

    Thread.Sleep(1000);  // to keep distance between cams
}


Task.WhenAll(tasks).GetAwaiter().GetResult();
