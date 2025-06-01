using NumberplateRecognition.Entities;
using OpenCvSharp;
using System.Text;
using System.Text.Json;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// It's an implementation of ILicensePlateReader which relies on an external REST service(probably written in python) that uses Pytorch native API and runs completely on CUDA.
    /// </summary>
    public class DtrReader : ILicensePlateReader
    {
        Size Shape { get; set; } = new Size(960, 640);

        private readonly string _modelPath;

        private readonly HttpClient _httpClient;

        /// <summary>
        /// This constructor gets http address of service for communication and processing frames.
        /// </summary>
        /// <param name="modelPath">AI model server http address</param>
        public DtrReader(string modelPath)
        {
            _modelPath = modelPath;
            _httpClient = new HttpClient();
        }

        public async Task<string> ReadPlate(Mat frame)
        {
            var image = frame.Clone();

            byte[][][] tensor = new byte[Shape.Height][][];

            for (int h = 0; h < Shape.Height; h++)
            {
                tensor[h] = new byte[Shape.Width][];
                for (int w = 0; w < Shape.Width; w++)
                {
                    var pixels = image.At<Vec3b>(h, w);
                    tensor[h][w] = [pixels.Item0, pixels.Item1, pixels.Item2];
                }
            }

            image.Dispose();

            var input = new { image = tensor, shape = new[] { Shape.Height, Shape.Width, 3 } };
            var jsonInput = JsonSerializer.Serialize(input);
            var content = new StringContent(jsonInput, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(_modelPath, content);

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine("Service Error: " + _modelPath);
                return "Error";
            }

            var jsonResponse = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<RecognitionResult>(jsonResponse);
            if (result?.Result == null)
            {
                Console.WriteLine("Service app internal error: " + _modelPath);
                return "Error";
            }

            if (result.Result == "NotFound")
            {
                Console.WriteLine("Plate Detection failed!");
                return "NotFound";
            }

            if (result.Result == "None")
            {
                Console.WriteLine("Text recognition failed!");
                return "None";
            }

            return result.Result;
        }
    }
}
