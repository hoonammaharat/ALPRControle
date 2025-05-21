using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Diagnostics;
using System.Xml.Linq;
using static System.Net.Mime.MediaTypeNames;

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

        public Model(string modelPath, string? executionProvider, Dictionary<string, string>? providerOption)
        {
            if (executionProvider != null)
            {
                var options = new SessionOptions();
                options.AppendExecutionProvider(executionProvider, providerOption);

                // OpenVino needs a special version of onnxruntime.dll, which is not copied in output dir automatically; copy this script in .csproj file:
                // < !-- < Target Name = "ReplaceOnnxRuntimeDLLForDirectML" AfterTargets = "Build" > < Message Text = "Replacing OnnxRuntime.dll with DirectML version. it's harmless and compatible, it can run model in simple mode; just other ExecutionProviders may need another special runtime" Importance = "high" /> < Copy SourceFiles = "C:\Users\Aseman Rasam\.nuget\packages\intel.ml.onnxruntime.directml\1.22.0\runtimes\win-x64\native\onnxruntime.dll" DestinationFiles = "C:\Users\Aseman Rasam\source\repos\NumberplateRecognition\NumberplateRecognition\bin\Debug\net9.0\runtimes\win-x64\native\onnxruntime.dll" OverwriteReadOnlyFiles = "true" SkipUnchangedFiles = "false" /> </ Target > -->

                // options.AppendExecutionProvider_DML();
                // DML needs a special version of onnxruntime.dll, which is not copied in output dir automatically; copy this script in .csproj file:
                // < !-- < Target Name = "ReplaceOnnxRuntimeDLLForDirectML" AfterTargets = "Build" > < Message Text = "Replacing OnnxRuntime.dll with DirectML version. it's harmless and compatible, it can run model in simple mode; just other ExecutionProviders may need another special runtime" Importance = "high" /> < Copy SourceFiles = "C:\Users\Aseman Rasam\.nuget\packages\microsoft.ml.onnxruntime.directml\1.22.0\runtimes\win-x64\native\onnxruntime.dll" DestinationFiles = "C:\Users\Aseman Rasam\source\repos\NumberplateRecognition\NumberplateRecognition\bin\Debug\net9.0\runtimes\win-x64\native\onnxruntime.dll" OverwriteReadOnlyFiles = "true" SkipUnchangedFiles = "false" /> </ Target > -->

                Session = new InferenceSession(modelPath, options);
            }
            Session = new InferenceSession(modelPath);
            Console.WriteLine("Model loaded successfully.\n");
        }

        public bool DetectTruck(Mat frame)
        {
            if (frame.Height != 640 || frame.Width != 960) throw new ArgumentException("Size is not correct!");

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


                var mostTruckScoreBoxes = new List<int>();
                for (int b = 0; b < result.Dimensions[2]; b++)
                {
                    if (result[0, 4, b] > 0.1 && result[0, 12, b] > 0.1)
                    {
                        mostTruckScoreBoxes.Add(b);
                    }

                    if (result[0, 4, b] > 0.4 && result[0, 12, b] > 0.4)
                    {
                        Console.WriteLine("truck found in box " + b.ToString() + " with:\nconfidence: " + result[0, 4, b].ToString() + "\nscore: " + result[0, 12, b].ToString() + "\n");
                        return true;
                    }
                }

                // For debugging pupose and tracing footprint:

                Console.WriteLine("most truck score boxes: \n");
                int roof = 0;
                foreach (int i in mostTruckScoreBoxes)
                {
                    Console.WriteLine(i.ToString() + ": conf = " + result[0, 4, i].ToString() + ", truck: " + result[0, 12, i].ToString() + ", train: " + result[0, 11, i].ToString());
                    if (roof >= 6) break;
                }
            }

            Console.WriteLine("truck not found\n");
            return false;
        }
    }
}
