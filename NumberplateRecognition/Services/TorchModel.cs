using System.Text;
using System.Text.Json;
using OpenCvSharp;
using NumberplateRecognition.Entities;

namespace NumberplateRecognition.Services
{
    public class TorchModel : ITruckDetectorModel
    {
        private readonly string _modelPath;

        private readonly HttpClient _httpClient;

        public TorchModel(string modelPath)
        {
            _modelPath = modelPath;
            _httpClient = new HttpClient();
        }

        public async Task<bool> DetectTruck(Mat frame)
        {
            if (frame.Height != 640 || frame.Width != 960) throw new ArgumentException("Size is not correct!");

            var image = frame.Clone();

            var tensor = new float[1, 3, 640, 960];

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

            var input = new { data = tensor, shape = new[] { 1, 3, frame.Height, frame.Width } };
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

            return true;
        }
    }
}
