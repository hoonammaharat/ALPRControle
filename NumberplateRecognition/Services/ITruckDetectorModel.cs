using OpenCvSharp;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// This is an interface for communication with AI models regarless of their backend and teck stack for lose coupling and stability.
    /// </summary>
    public interface ITruckDetectorModel
    {
        /// <summary>
        /// DetectTruck method gets a frame as a Mat object with RGB color code and returns truck existence status.
        /// </summary>
        /// <param name="frame">An OpenCvSharp4 Mat object</param>
        /// <returns>If truck found or not</returns>
        Task<bool> DetectTruck(Mat frame);
    }
}
