using System.Text.Json;
using System.Net.Http.Headers;
using OpenCvSharp;
using NumberplateRecognition.Entities;
using Serilog;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// It's an implementation of ITruckDetectorModel which relies on an external REST service(probably written in python) that uses Pytorch native API and runs completely on CUDA.
    /// </summary>
    public class TorchTruckDetectionModel : ITruckDetectorModel, IDisposable
    {
        Size Shape { get; set; }

        private readonly string _modelPath;

        private readonly HttpClient _httpClient;

        /// <summary>
        /// This constructor gets http address of service for communication and processing frames.
        /// </summary>
        /// <param name="modelPath">AI model server http address</param>
        public TorchTruckDetectionModel(string modelPath)
        {
            _modelPath = modelPath;
            _httpClient = new HttpClient();
        }

        public void Dispose()
        {
            _httpClient.Dispose();
        }

        public async Task<bool> DetectTruck(Mat frame)
        {
            try
            {
                using var image = frame.Clone();
                Shape = new Size(image.Width / 32 * 32, image.Height / 32 * 32);

                byte[] flat = new byte[1 * 3 * Shape.Height * Shape.Width];

                for (int h = 0; h < Shape.Height; h++) // image to tensor to flat array conversion
                {
                    for (int w = 0; w < Shape.Width; w++)
                    {
                        var pixel = image.At<Vec3b>(h, w);
                        flat[(0 * Shape.Height + h) * Shape.Width + w] = pixel.Item2;  // tensor byte[1, channel, height, width] -> tensor[0, c, h, w] = image.At<Vec3b>(h, w)[c]  or  (channel - c) for convert to RGB
                        flat[(1 * Shape.Height + h) * Shape.Width + w] = pixel.Item1;
                        flat[(2 * Shape.Height + h) * Shape.Width + w] = pixel.Item0;  // flat index = ((b * C + c) * H + h) * W + w;  b = 0  ->  (c * H + h) * W + w
                    }
                }

                using var content = new ByteArrayContent(flat);
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                content.Headers.Add("Shape", $"1,3,{Shape.Height},{Shape.Width}");

                using var response = await _httpClient.PostAsync(_modelPath, content);

                if (!response.IsSuccessStatusCode)
                {
                    Console.WriteLine("Service Error: " + _modelPath);
                    return false;
                }

                var jsonResponse = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<DetectionResult>(jsonResponse);
                if (result?.Result == null)
                {
                    Console.WriteLine("Service app internal error: " + _modelPath);
                    return false;
                }

                if (result.Result == "true") return true;
                return false;
            }

            catch (Exception ex)
            {
                Log.Error(ex, "Detection Truck Error");
                return false;
            }
        }
    }
}
