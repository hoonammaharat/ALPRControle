using NumberplateRecognition;
using OpenCvSharp;
using System.IO;
using System.Threading.Channels;


List<string> inputCamURLs = [ "rtsp://:8554/test" ];
List<string> outputCamURLs = [];
List<Task> tasks = [];


// Manual and direct path for debugging Model execution without concurrency complexities:

List<string> pathes = new List<string>() { "E:\\a.jpg", "E:\\b.jpg", "E:\\c.jpg", "E:\\e.jpg", "E:\\f.jpg" };
var model = new Model("E:\\yolo11s.onnx");
foreach (var path in pathes)
{
    Console.WriteLine("File: " + path);
    var frame = Cv2.ImRead(path);
    Cv2.CvtColor(frame, frame, ColorConversionCodes.BGR2RGB);
    model.DetectTruck(frame);
}


// Actual Algoritm for running app properly and concurrently:

//for (int x = 0; x < inputCamURLs.Count; x++)
//{
//    var channel = Channel.CreateUnbounded<Record>();

//    var t1 = Task.Run(async () =>
//    {
//        // Debugging model execution in concurrent mode without network streaming execution:
//
//        /*foreach(var path in pathes)
//        {
//            Console.WriteLine("File: " + path + "\n");
//            var frame = Cv2.ImRead(path);
//            var record = new Record(frame.Clone(), 1);
//            frame.Dispose();
//            await channel.Writer.WriteAsync(record);
//            await Task.Delay(450);
//        }*/

//        /*var capture = new VideoCapture(inputCamURLs[x]);
//        if (!capture.IsOpened())
//        {
//            Console.WriteLine("Connection failed!");
//        }

//        while (true)
//        {
//            bool success = capture.Read(frame);
//            if (success && !frame.Empty())
//            {
//                var record = new Record(frame.Clone(), x);
//                await channel.Writer.WriteAsync(record);
//                await Task.Delay(450);
//            }
//            else
//            {
//                Console.WriteLine("Reading stream failed or frame is empty!");
//            }
//        }*/
//    });
//    tasks.Add(t1);

//    var model = new Model("E:\\yolo11n.onnx");

//    var t2 = Task.Run(async () =>
//    {
//        while (true)
//        {
//            var record = await channel.Reader.ReadAsync();
//            Cv2.CvtColor(record.Frame, record.Frame, ColorConversionCodes.BGR2RGB);
//            model.DetectTruck(record.Frame);
//        }
//    });
//    tasks.Add(t2);
//}

//Task.WhenAll(tasks).GetAwaiter().GetResult();
