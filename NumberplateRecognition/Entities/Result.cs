namespace NumberplateRecognition.Entities
{
    /// <summary>
    /// This entity stores result of TruckDetection gotten from external services.
    /// </summary>
    public class Result
    {
        public int[]? Shape { get; set; }

        public float[][][]? Data { get; set; }
    }
}
