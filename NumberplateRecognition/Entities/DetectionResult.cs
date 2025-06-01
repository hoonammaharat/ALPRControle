namespace NumberplateRecognition.Entities
{
    /// <summary>
    /// This entity stores result of Truck Detection gotten from external services.
    /// </summary>
    public class DetectionResult
    {
        public int[]? Shape { get; set; }

        public float[][][]? Output { get; set; }
    }
}
