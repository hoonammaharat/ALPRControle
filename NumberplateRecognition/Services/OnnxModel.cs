using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;


namespace NumberplateRecognition.Services
{
    /// <summary>
    /// It's an implementation of ITruckDetectorModel which relies on an internal ONNX Runtime session without need of external services and Pytorch native APIs.
    /// </summary>
    public class OnnxModel : ITruckDetectorModel
    {
        public (int, int) Shape { get; set; } = (640, 960);

        private readonly InferenceSession? _session;

        /// <summary>
        /// This constructor is used to start a ort session from given model file and specified execution provider and options.
        /// </summary>
        /// <param name="modelPath">The path to model file with onnx format</param>
        /// <param name="executionProvider">ONNX Runtime backend for executing algoritm; by default it runs algorithm on CPU, CUDA is recommended</param>
        /// <param name="providerOption">Extra options and settings which may be necessary for an EP</param>
        public OnnxModel(string modelPath, string? executionProvider = null, Dictionary<string, string>? providerOption = null)
        {
            if (executionProvider != null)
            {
                var options = new SessionOptions();
                if (executionProvider == "CUDA")
                {
                    options.AppendExecutionProvider_CUDA(0);
                    // CUDA needs a special version of onnxruntime.dll, which is not copied in output dir automatically; copy this script in .csproj file:
                    // < !-- < Target Name = "ReplaceOnnxRuntimeDLLForCUDA" AfterTargets = "Build" > < Message Text = "Replacing OnnxRuntime.dll with CUDA version. it's harmless and compatible, it can run model in simple mode; just other ExecutionProviders may need another special runtime" Importance = "high" /> < Copy SourceFiles = "C:\Users\Arian\.nuget\packages\microsoft.ml.onnxruntime.gpu.windows\1.22.0\runtimes\win-x64\native\onnxruntime.dll" DestinationFiles = "E:\Projects\NumberplateRecognition\NumberplateRecognition\bin\Debug\net9.0\runtimes\win-x64\native\onnxruntime.dll" OverwriteReadOnlyFiles = "true" SkipUnchangedFiles = "false" /> </ Target >
                    _session = new InferenceSession(modelPath, options);
                }
                else if (executionProvider == "DirectML")
                {
                    // options.AppendExecutionProvider_DML();
                    // DML needs a special version of onnxruntime.dll, which is not copied in output dir automatically; copy this script in .csproj file:
                    // < !-- < Target Name = "ReplaceOnnxRuntimeDLLForDirectML" AfterTargets = "Build" > < Message Text = "Replacing OnnxRuntime.dll with DirectML version. it's harmless and compatible, it can run model in simple mode; just other ExecutionProviders may need another special runtime" Importance = "high" /> < Copy SourceFiles = "C:\Users\Aseman Rasam\.nuget\packages\microsoft.ml.onnxruntime.directml\1.22.0\runtimes\win-x64\native\onnxruntime.dll" DestinationFiles = "C:\Users\Aseman Rasam\source\repos\NumberplateRecognition\NumberplateRecognition\bin\Debug\net9.0\runtimes\win-x64\native\onnxruntime.dll" OverwriteReadOnlyFiles = "true" SkipUnchangedFiles = "false" /> </ Target > -->
                    // _session = new InferenceSession(modelPath, options);
                }
                else if (executionProvider == "OpenVINO")
                {
                    options.AppendExecutionProvider(executionProvider, providerOption);
                    // OpenVino needs a special version of onnxruntime.dll, which is not copied in output dir automatically; copy this script in .csproj file:
                    // < !-- < Target Name = "ReplaceOnnxRuntimeDLLForDirectML" AfterTargets = "Build" > < Message Text = "Replacing OnnxRuntime.dll with DirectML version. it's harmless and compatible, it can run model in simple mode; just other ExecutionProviders may need another special runtime" Importance = "high" /> < Copy SourceFiles = "C:\Users\Aseman Rasam\.nuget\packages\intel.ml.onnxruntime.directml\1.22.0\runtimes\win-x64\native\onnxruntime.dll" DestinationFiles = "C:\Users\Aseman Rasam\source\repos\NumberplateRecognition\NumberplateRecognition\bin\Debug\net9.0\runtimes\win-x64\native\onnxruntime.dll" OverwriteReadOnlyFiles = "true" SkipUnchangedFiles = "false" /> </ Target > -->
                    _session = new InferenceSession(modelPath, options);
                }
            }
            else _session = new InferenceSession(modelPath);

            Console.WriteLine("Model loaded successfully.\n");
        }

        public Task<bool> DetectTruck(Mat frame)
        {
            if (frame.Height != Shape.Item1 || frame.Width != Shape.Item2) throw new ArgumentException("Size is not correct!");

            var image = frame.Clone();

            var tensor = new DenseTensor<float>([1, 3, Shape.Item1, Shape.Item2]);

            for (int h = 0; h < Shape.Item1; h++)  // image to tensor conversion
            {
                for (int w = 0; w < Shape.Item2; w++)
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
            using (var output = _session!.Run(input))
            {
                timer.Stop();
                Console.WriteLine("Model Latency: " + timer.ElapsedMilliseconds.ToString());  // Model Latency

                var result = output.First().AsTensor<float>();  // result is a disposable list of OnnxNamedValue, but you has only one output, we get and convert it to Tensor

                for (int b = 0; b < result.Dimensions[1]; b++)
                {
                    if (result[0, b, 4] > 0.6 && result[0, b, 5] == 7)
                    {
                        Console.WriteLine("truck found in box " + b.ToString() + " with confidence: " + result[0, b, 4].ToString() + "\n\n");
                        return Task.FromResult(true);
                    }
                }
            }

            Console.WriteLine("truck not found\n\n");
            return Task.FromResult(false);
        }
    }
}
