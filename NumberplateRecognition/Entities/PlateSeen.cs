namespace NumberplateRecognition.Entities;

public class PlateSeen
{
    public required string Plate { get; set; }
    
    public DateTime LastSeen { get; set; }
}
