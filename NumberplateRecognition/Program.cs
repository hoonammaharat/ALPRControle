using NumberplateRecognition;
using OpenCvSharp;
using System.Threading.Channels;


List<string> inputCamURLs = [ "rtsp://:8554/test" ];
List<string> outputCamURLs = [];
List<Task> tasks = [];

var model = new Model("E:\\yolo11n.onnx");
Thread.Sleep(5000);
List<string> pathes = new List<string>() { "E:\\a.jpg", "E:\\b.jpg", "E:\\c.jpg", "E:\\d.jpg", "E:\\e.jpg" };


//for (int x = 0; x < inputCamURLs.Length; x++)
//{
    var channel = Channel.CreateUnbounded<Record>();

    // var t1 = Task.Run(async () =>
    Thread t1 = new Thread(async () =>
    {
        foreach(var path in pathes)
        {
            var frame = Cv2.ImRead(path);
            var record = new Record(frame.Clone(), 1);
        }

        //var capture = new VideoCapture(inputCamURLs[x]);
        //if (!capture.IsOpened())
        //{
        //    Console.WriteLine("Connection failed!");
        //}

        //while (true)
        //{
            //bool success = capture.Read(frame);
            //if (success && !frame.Empty())
            //{
            //    var record = new Record(frame.Clone(), x);
            //    await channel.Writer.WriteAsync(record);
            //    await Task.Delay(450);
            //}
            //else
            //{
            //    Console.WriteLine("Reading stream failed or frame is empty!");
            //}
        //}
    });
    //tasks.Add(t1);

    //var t2 = Task.Run(async () =>
    Thread t2 = new Thread(async () =>
    {
        var model = new Model("E:\\yolo11n.onnx");

        while (true)
        {
            var record = await channel.Reader.ReadAsync();
            Cv2.CvtColor(record.Frame, record.Frame, ColorConversionCodes.BGR2RGB);
            model.DetectTruck(record.Frame);
        }
    });
    //tasks.add(t2);
//}

// Task.WhenAll(tasks).GetAwaiter().GetResult();

t1.Start();
t2.Start();

t1.Join();
t2.Join();
