using System.Threading.Channels;
using OpenCvSharp;

namespace NumberplateRecognition
{
    public class Program
    {
        public static void Main()
        {
            string[] inputCamURLs = { };
            string[] outputCamURLs = { };

            for (int x = 1; x < inputCamURLs.Length; x++)
            {
                var channel = Channel.CreateUnbounded<Record>();

                Task.Run(async () =>
                {
                    var capture = new VideoCapture(inputCamURLs[x]);
                    if (!capture.IsOpened())
                    {
                        Console.WriteLine("Connection failed!");
                    }

                    var frame = new Mat();

                    while (true)
                    {
                        bool success = capture.Read(frame);
                        if (success && !frame.Empty())
                        {
                            var record = new Record(frame.Clone(), x);
                            await Task.Delay(450);
                        }
                        else
                        {
                            Console.WriteLine("Frame is empty!");
                        }
                    }
                });

                Task.Run(() =>
                {

                });
            }


        }
    }
}
