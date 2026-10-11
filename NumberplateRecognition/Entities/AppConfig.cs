namespace NumberplateRecognition.Entities;

public class AppConfig
{
    public List<ControlCamera> Cameras { get; set; } = [];

    public required string DetectionModelType { get; set; }
    
    public required string DetectionModelPath { get; set; }
    
    public required string RecognitionModelPath { get; set; }
    
    public required string ApiPath { get; set; }
    
    public required string MinimumLogLevel { get; set; }
    
    public required string Seq { get; set; }
}