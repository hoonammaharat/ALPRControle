using System.Text;
using System.Text.Json;
using OpenCvSharp;
using NumberplateRecognition.Entities;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// It's an implementation of ITruckDetectorModel which relies on an external REST service(probably written in python) that uses Pytorch native API and runs completely on CUDA.
    /// </summary>
    public class TorchModel : ITruckDetectorModel
    {
        (int, int) Shape { get; set; } = (640, 960);

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
            if (frame.Height != Shape.Item1 || frame.Width != Shape.Item2) throw new ArgumentException("Size is not correct!");

            var image = frame.Clone();

            float[][][][] tensor = new float[1][][][];
            tensor[0] = new float[3][][];

            for (int c = 0; c < 3; c++)
            {
                tensor[0][c] = new float[Shape.Item1][];
                for (int h = 0; h < Shape.Item1; h++)
                {
                    tensor[0][c][h] = new float[Shape.Item2];
                    for (int w = 0; w < Shape.Item2; w++)
                    {
                        tensor[0][c][h][w] = image.At<Vec3b>(h, w)[c];
                    }
                }
            }

            image.Dispose();

            var input = new { data = tensor, shape = new[] { 1, 3, Shape.Item1, Shape.Item2 } };
            var jsonInput = JsonSerializer.Serialize(input);
            var content = new StringContent(jsonInput, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(_modelPath, content);

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine("Operation failed!");
                return false;
            }

            var jsonResponse = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<Result>(jsonResponse);
            if (result?.Data == null)
            {
                Console.WriteLine("Operation failed!");
                return false;
            }

            for (int b = 0; b < result.Data[1].Length; b++)
            {
                if (result.Data[0][b][4] > 0.6 && result.Data[0][b][5] == 7)
                {
                    Console.WriteLine("truck found in box " + b.ToString() + " with confidence: " + result.Data[0][b][4].ToString() + "\n\n");
                    return true;
                }
            }

            return false;
        }
    }
}
