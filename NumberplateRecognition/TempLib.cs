using System.Diagnostics;
using System.Runtime.CompilerServices;
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
            Console.WriteLine("Model loaded successfully.\n");
        }

        public bool DetectTruck(Mat frame)
        {
            if (frame.Height != 640 || frame.Width != 960) throw new ArgumentException("size is not correct!");

            var image = frame.Clone();

            var tensor = new DenseTensor<float>([1, 3, 640, 960]);

            for (int h = 0; h < 640; h++)  // image to tensor conversion
            {
                for (int w = 0; w < 960; w++)
                {
                    var pixel = image.At<Vec3b>(h, w);
                    tensor[0, 0, h, w] = pixel.Item0;
                    tensor[0, 1, h, w] = pixel.Item1;
                    tensor[0, 2, h, w] = pixel.Item2;
                }
            }

            image.Dispose();

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };

            var timer = Stopwatch.StartNew();
            using (var output = Session.Run(input))
            {
                timer.Stop();
                Console.WriteLine("Model Latency: " + timer.ElapsedMilliseconds.ToString() + "\n");  // Model Latency

                var result = output.First().AsTensor<float>();  // result is a disposable list of OnnxNamedValue, but you has only one output, we get and convert it to Tensor

                int max = 0;
                for (int b = 0; b < result.Dimensions[2]; b++)
                {
                    if (result[0, 4, b] > max) max = b;

                    if (result[0, 4, b] > 0.5 && result[0, 12, b] > 0.5)
                    {
                        Console.WriteLine("truck found in box " + b.ToString() + " with:\nconfidence: " + result[0, 4, b].ToString() + "\nscore: " + result[0, 12, b].ToString() + "\n");
                        return true;
                    }
                }

                var classScores = new List<(int, float)>();
                for (int i = 5; i < result.Dimensions[1]; i++)
                {
                    classScores.Add((i, result[0, i, max]));
                }

                var sorted = classScores.OrderByDescending(x => x.Item2);

                Console.WriteLine("max confident box: " + max.ToString() + "\n");
                foreach (var s in sorted) { Console.WriteLine(s.Item1.ToString() + ": " + s.Item2.ToString() + "\n"); }
            }


            Console.WriteLine("truck not found\n");
            return false;
        }
    }
}
