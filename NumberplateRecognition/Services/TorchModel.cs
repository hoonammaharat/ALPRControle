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

                float[][][][] tensor = new float[1][][][];
                tensor[0] = new float[3][][];

                for (int c = 0; c < 3; c++)
                {
                    tensor[0][c] = new float[Shape.Height][];
                    for (int h = 0; h < Shape.Height; h++)
                    {
                        tensor[0][c][h] = new float[Shape.Width];
                        for (int w = 0; w < Shape.Width; w++)
                        {
                            tensor[0][c][h][w] = image.At<Vec3b>(h, w)[2 - c] / 255.0f;
                        }
                    }
                }

                image.Dispose();

                var input = new { image = tensor, shape = new[] { 1, 3, Shape.Height, Shape.Width } };
                var jsonInput = JsonSerializer.Serialize(input);
                var content = new StringContent(jsonInput, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync(_modelPath, content);

                if (!response.IsSuccessStatusCode)
                {
                    Console.WriteLine("Service Error: " + _modelPath);
                    return false;
                }

                var jsonResponse = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<DetectionResult>(jsonResponse);
                if (result?.Output == null)
                {
                    Console.WriteLine("Service app internal error: " + _modelPath);
                    return false;
                }

                for (int b = 0; b < result.Output[0].Length; b++)
                {
                    if (result.Output[0][b][4] > 0.6 && result.Output[0][b][5] == 7)
                    {
                        return true;
                    }
                }

                return false;
            }

            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return false;
            }
        }
    }
}
