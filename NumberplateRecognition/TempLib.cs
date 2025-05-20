using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace NumberplateRecognition
{
    public class Record
    {
        public Mat Frame { get; set; }

        public int CameraID { get; set; }

        public Record(Mat frame, int id)
        {
            Frame = frame;
            CameraID = id;
        }

        public void Dispose()
        {
            Frame?.Dispose();
        }
    }

    public class Model
    {
        public InferenceSession Session { get; set; }

        public Model(string modelPath)
        {
            Session = new InferenceSession(modelPath);
            Console.WriteLine("Model loaded correctly.");
        }

        public bool DetectTruck(Mat frame)
        {
            var image = new Mat();
            Cv2.Resize(frame, image, new Size(960, 640));

            var tensor = new DenseTensor<float>([1, 3, 640, 960]);

            for (int h = 0; h < 640; h++)
            {
                for (int w = 0; w < 960; w++)
                {
                    var pixel = image.At<Vec3b>(h, w);
                    tensor[0, 0, h, w] = pixel.Item0 / 255.0f;
                    tensor[0, 1, h, w] = pixel.Item1 / 255.0f;
                    tensor[0, 2, h, w] = pixel.Item2 / 255.0f;
                }
            }

            image.Dispose();
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

            var timer = Stopwatch.StartNew();
            using (var output = Session.Run(input))
            {
                timer.Stop();
                Console.WriteLine(timer.ElapsedMilliseconds);
                var result = output.First().AsTensor<float>();
                for (int b = 0; b < result.Dimensions[1]; b++)
                {
                    if (result[0, b, 4] > 0.5 && result[0, b, 12] > 0.55)
                    {
                        Console.WriteLine("truck found");
                        return true;
                    }
                }
            }

            Console.WriteLine("truck not found");
            return false;
        }
    }
}
