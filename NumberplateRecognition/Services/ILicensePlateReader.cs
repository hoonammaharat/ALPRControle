using OpenCvSharp;

namespace NumberplateRecognition.Services
{
    /// <summary>
    /// This is an interface for communication with AI models regardless of their backend and tech stack for lose coupling and stability.
    /// </summary>
    public interface ILicensePlateReader
    {
        /// <summary>
        /// Read method gets a frame as a Mat object with RGB color code and returns license plate text or status.
        /// </summary>
        /// <param name="frame">An OpenCvSharp4 Mat object</param>
        /// <returns>Plate text or status(plus confidence score): ServiceError, NotFound(plate in frame), None(unable to read), ClientError</returns>
        Task<(string, float)> ReadPlate(Mat frame);
    }
}
