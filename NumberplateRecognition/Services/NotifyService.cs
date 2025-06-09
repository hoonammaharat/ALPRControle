using System.Text;
using System.Text.Json;
using NumberplateRecognition.Entities;
using OpenCvSharp;
using Serilog;

namespace NumberplateRecognition.Services
{
    public class NotifyService : IDisposable
    {
        public string ApiPath { get; }

        private readonly HttpClient _httpClient;

        public NotifyService(string path)
        {
            ApiPath = path;
            _httpClient = new HttpClient();
        }

        public void Dispose()
        {
            _httpClient.Dispose();
        }

        public async Task<bool> NotifyApi(int id, string ip, string name, DateTime now, string? text, Mat image, float? conf = 0)
        {
            try
            {
                bool fail = false;
                Plate plate = new Plate();

                if (text?.Length == 8)
                {
                    var d = text.Substring(0, 2);
                    var a = text.Substring(2, 1);
                    var t = text.Substring(3, 3);
                    var ir = text.Substring(6, 2);

                    try
                    {
                        plate.DoualPart = Convert.ToInt32(d);
                        plate.AlphaPart = a;
                        plate.Triplepart = Convert.ToInt32(t);
                        plate.IranCode = Convert.ToInt32(ir);
                    }
                    catch
                    {
                        fail = true;
                    }
                }
                else fail = true;

                Cv2.ImEncode(".jpg", image, out byte[] imageBytes);

                var data = new
                {
                    CameraId = id, Ip = ip, Name = name, DateTime = now, OrginalPlate = text ?? "null",
                    ConfidenceFactor = conf,
                    PlateNotDetected = fail, Plate = plate, image = Convert.ToBase64String(imageBytes)
                };

                string json = JsonSerializer.Serialize(data);

                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = await _httpClient.PostAsync(ApiPath, content);
                if (!response.IsSuccessStatusCode)
                {
                    Log.Error("Response Error {StatusCode} while attempting to send detected plate in camera #{id}: {val}", response.StatusCode.ToString(), id, ip);
                    return false;
                }

                return true;
            }

            catch (Exception ex)
            {
                Log.Error(ex, "An error occured while attempting to send detected plate in camera #{id}: {val}", id, ip);
                return false;
            }
        }
    }
}
