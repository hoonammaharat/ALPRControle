using NumberplateRecognition.Entities;
using OpenCvSharp;
using System.Net.Http.Headers;
using System.Text.Json;
using Serilog;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// It's an implementation of ILicensePlateReader which relies on an external REST service(probably written in python) that uses Deep Text Recognition models in Pytorch native API and runs completely on CUDA.
    /// </summary>
    public class DtrReader : ILicensePlateReader, IDisposable
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

        public void Dispose()
        {
            _httpClient.Dispose();
        }

        public async Task<(string, float)> ReadPlate(Mat frame)
        {
            try
            {
                Shape = new Size(frame.Width / 32 * 32, frame.Height / 32 * 32);
                using var image = new Mat();
                Cv2.Resize(frame, image, Shape);

                byte[] flat = new byte[Shape.Height * Shape.Width * 3];

                System.Runtime.InteropServices.Marshal.Copy(image.Data, flat, 0, flat.Length);

                using var content = new ByteArrayContent(flat);
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                content.Headers.Add("Shape", $"{Shape.Height},{Shape.Width},3");

                using var response = await _httpClient.PostAsync(_modelPath, content);

                if (!response.IsSuccessStatusCode)
                {
                    return ("ServiceError", 0);
                }

                var jsonResponse = await response.Content.ReadAsStringAsync();
                Log.Information(jsonResponse);
                var result = JsonSerializer.Deserialize<RecognitionResult>(jsonResponse);
                if (result?.Result == null || result?.Confidence == null)
                {
                    return ("InternalServiceError", 0);
                }

                if (result.Result == "NotFound")
                {
                    Log.Error("Plate not found in detected image!");
                    return ("ClientError", 0);
                }

                return (result.Result, result.Confidence.Value);
            }

            catch (Exception ex)
            {
                Log.Error(ex, "Reading Plate Error");
                return ("ClientError", 0);
            }
        }
    }
}
