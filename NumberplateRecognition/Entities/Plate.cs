namespace NumberplateRecognition.Entities
{
    public class Plate
    {
        public int DoualPart { get; set; }

        public string AlphaPart { get; set; }
        
        public int Triplepart { get; set; }

        public int IranCode { get; set; }

        public Plate()
        {
            DoualPart = 0;
            AlphaPart = string.Empty;
            Triplepart = 0;
            IranCode = 0;
        }

        public override string ToString()
        {
            return $"{IranCode} {Triplepart}-{AlphaPart}-{DoualPart} ";
        }
    }
}
