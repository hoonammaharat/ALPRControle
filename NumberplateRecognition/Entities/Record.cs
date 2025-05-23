using OpenCvSharp;

namespace NumberplateRecognition.Entities
{
    /// <summary>
    /// This entity contains gotten frame from cameras and cameras' id for context, method and thread independency.
    /// Considering it has unmanaged native memory last owner of this entity's object should dispose it.
    /// </summary>
    public class Record : IDisposable
    {
        public Mat Frame { get; set; }

        public int CameraID { get; set; }

        public Record(Mat frame, int id)
        {
            Frame = frame;
            CameraID = id;
        }

        public void Dispose()
        {
            Frame?.Dispose();
        }
    }
}
