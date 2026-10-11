$workDir = "PlateReaderPath"

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$workDir'; python plate_recognition_server_oldmodel.py"
)

$count = [int](Read-Host "Enter count")

for ($i = 0; $i -lt $count; $i++) {
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$workDir'; python plate_detection_server.py $i"
    )
}