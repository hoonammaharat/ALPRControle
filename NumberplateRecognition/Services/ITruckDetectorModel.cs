using OpenCvSharp;

namespace NumberplateRecognition.Services
{
    public interface ITruckDetectorModel
    {
        bool DetectTruck(Mat frame);
    }
}
