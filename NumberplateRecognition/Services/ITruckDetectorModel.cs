using OpenCvSharp;

namespace NumberplateRecognition.Services
{
    public interface ITruckDetectorModel
    {
        Task<bool> DetectTruck(Mat frame);
    }
}
