using System.Text.Json;
using System.Net.Http.Headers;
using OpenCvSharp;
using NumberplateRecognition.Entities;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// It's an implementation of ITruckDetectorModel which relies on an external REST service(probably written in python) that uses Pytorch native API and runs completely on CUDA.
    /// </summary>
    public class TorchModel : ITruckDetectorModel
    {
        Size Shape { get; set; }

        private readonly string _modelPath;

        private readonly HttpClient _httpClient;

        /// <summary>
        /// This constructor gets http address of service for communication and processing frames.
        /// </summary>
        /// <param name="modelPath">AI model server http address</param>
        public TorchModel(string modelPath)
        {
            _modelPath = modelPath;
            _httpClient = new HttpClient();
        }

        public async Task<bool> DetectTruck(Mat frame)
        {
            try
            {
                var image = frame.Clone();
                Shape = new Size(image.Width / 32 * 32, image.Height / 32 * 32);

                byte[,,,] tensor = new byte[1, 3, Shape.Height, Shape.Width];

                for (int h = 0; h < Shape.Height; h++) // image to tensor conversion
                {
                    for (int w = 0; w < Shape.Width; w++)
                    {
                        var pixel = image.At<Vec3b>(h, w);
                        tensor[0, 0, h, w] = pixel.Item2;
                        tensor[0, 1, h, w] = pixel.Item1;
                        tensor[0, 2, h, w] = pixel.Item0;
                    }
                }

                image.Dispose();

                byte[] flat = new byte[tensor.Length];
                Buffer.BlockCopy(tensor, 0, flat, 0, flat.Length);

                var content = new ByteArrayContent(flat);
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                content.Headers.Add("Shape", $"1,3,{Shape.Height},{Shape.Width}");

                var response = await _httpClient.PostAsync(_modelPath, content);

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

                else return false;
            }

            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return false;
            }
        }
    }
}
