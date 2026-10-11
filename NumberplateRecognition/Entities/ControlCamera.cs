namespace NumberplateRecognition.Entities;

public class ControlCamera
{
    public int Id { get; set; }
    
    public required string Name { get; set; }
    
    public required string Username { get; set; }
    
    public required string Password { get; set; }
    
    public required string Address { get; set; }

    public required string Path { get; set; } = "";
    
    public bool IsActive { get; set; }

    private const string Protocol = "rtsp";

    public string Url => $"{Protocol}://{Username}:{Password}@{Address}/{Path}";
}
