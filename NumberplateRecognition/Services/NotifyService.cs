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
        
        public string DefaultIranCode { get; }

        private readonly HttpClient _httpClient;

        public NotifyService(string path, string defaultIranCode)
        {
            ApiPath = path;
            DefaultIranCode = defaultIranCode;
            _httpClient = new HttpClient();
        }

        public void Dispose()
        {
            _httpClient.Dispose();
        }

        // public async Task<bool> NotifyApi(int id, string ip, string name, DateTime now, string? text, Mat image, float? conf = 0)
        // {
        //     try
        //     {
        //         bool fail = false;
        //         Plate plate = new Plate();
        //
        //         if (text?.Length >= 6)
        //         {
        //             var d = text.Substring(0, 2);
        //             var a = text.Substring(2, 1);
        //             var t = text.Substring(3, 3);
        //             var ir = (text.Length == 8) ? text.Substring(6, 2) : DefaultIranCode;
        //
        //             try
        //             {
        //                 plate.DoualPart = Convert.ToInt32(d);
        //                 plate.AlphaPart = (a == "j") ? "e" : a;
        //                 plate.IranCode = Convert.ToInt32(ir);
        //             }
        //             catch
        //             {
        //                 fail = true;
        //             }
        //             try
        //             {
        //                 plate.Triplepart = Convert.ToInt32(t);
        //             }
        //             catch
        //             {
        //                 fail = true;
        //             }
        //         }
        //         else fail = true;
        //
                // if (plate.DoualPart == 0 && plate.Triplepart == 0) return false;
                //
                // Cv2.ImEncode(".jpg", image, out byte[] imageBytes);
                //
                // var data = new
                // {
                //     CameraId = id, Ip = ip, Name = name, DateTime = now, OrginalPlate = text ?? "null",
                //     ConfidenceFactor = conf,
                //     PlateNotDetected = fail, Plate = plate, image = Convert.ToBase64String(imageBytes)
                // };
                //
                // string json = JsonSerializer.Serialize(data);
                //
                // using var content = new StringContent(json, Encoding.UTF8, "application/json");
                // using var response = await _httpClient.PostAsync(ApiPath, content);
                // if (!response.IsSuccessStatusCode)
                // {
                //     Log.Error("Response Error {StatusCode} while attempting to send detected plate in camera #{id}: {val}", response.StatusCode.ToString(), id, ip);
                //     return false;
                // }
                //
                // return true;
        //     }
        //
        //     catch (Exception ex)
        //     {
        //         Log.Error(ex, "An error occured while attempting to send detected plate in camera #{id}: {val}", id, ip);
        //         return false;
        //     }
        // }

        public async Task<bool> NotifyApi(int id, string ip, string name, DateTime now, string text, Mat image, float? conf = 0)
        {
            bool fail = false;
            
            Plate plate = new Plate();

            string? alphaPart = text.FirstOrDefault(a => !char.IsDigit(a)).ToString() ?? null;
            
            if (!string.IsNullOrEmpty(alphaPart))
            {
                plate.AlphaPart = alphaPart;
                
                int alphaPartIndex = text.IndexOf(alphaPart);
                plate.DoualPart = alphaPartIndex == 0 ? 0 : Convert.ToInt32(text[..alphaPartIndex]);

                if (alphaPartIndex + 3 < text.Length)
                    plate.Triplepart = Convert.ToInt32(text.Substring(alphaPartIndex + 1, 3));
                else fail = true;

                if (alphaPartIndex + 4 < text.Length)
                    plate.IranCode =
                        Convert.ToInt32(text.Substring(alphaPartIndex + 4, text.Length - alphaPartIndex - 4));
                else fail = true;
            }
            else
            {
                fail = true;
                
                if (text.Length >= 2) plate.DoualPart = Convert.ToInt32(text[..2]);

                if (text.Length >= 5) plate.Triplepart = Convert.ToInt32(text.Substring(2, 3));

                if (text.Length >= 7) plate.IranCode = Convert.ToInt32(text.Substring(6, text.Length - 6));
            }

            if (plate.IranCode < 10) plate.IranCode = Convert.ToInt32(DefaultIranCode);
            
            Cv2.ImEncode(".jpg", image, out byte[] imageBytes);
            
            var data = new
            {
                CameraId = id, Ip = ip, Name = name, DateTime = now, OrginalPlate = text,
                ConfidenceFactor = conf,
                PlateNotDetected = fail, Plate = plate, image = Convert.ToBase64String(imageBytes)
            };
            
            string json = JsonSerializer.Serialize(data);
            
            using var content = new StringContent(json, Encoding.UTF8, "application/json");
            try
            {
                using var response = await _httpClient.PostAsync(ApiPath, content);
                if (!response.IsSuccessStatusCode)
                {
                    Log.Error(
                        "Response Error {StatusCode} while attempting to send detected plate in camera #{id}: {val}",
                        response.StatusCode.ToString(), id, ip);
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
