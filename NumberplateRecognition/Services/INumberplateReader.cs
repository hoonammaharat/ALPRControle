using OpenCvSharp;

namespace NumberplateRecognition.Services
{
    public interface INumberplateReader
    {
        string ReadNumberplate(Mat frame);
    }
}
