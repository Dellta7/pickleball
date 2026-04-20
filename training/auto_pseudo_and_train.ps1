Param(
  [string]$Glob = "*.mp4",
  [double]$SampleFps = 6.0,
  [int]$Epochs = 60,
  [int]$Batch = 8,
  [int]$ImgSz = 640,
  [string]$Model = "yolov8n.pt",
  [string]$Name = "pickleball",
  [string]$Device = "0",
  [double]$ValRatio = 0.1,
  [int]$MaxFramesPerVideo = 0,
  [double]$BallConf = 0.05
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) {
  throw "Không thấy venv tại $Py. Hãy tạo venv trước hoặc dùng đúng workspace." 
}

function Test-CudaOp {
  & $Py -c "import torch; x=torch.randn((256,256), device='cuda'); y=x@x; print('cuda_ok')" | Out-Null
  return ($LASTEXITCODE -eq 0)
}

Write-Host "[1/3] Đảm bảo Torch CUDA chạy được (tự fallback nightly nếu cần)..."
& $Py -m pip install -U pip | Out-Null

$torchOk = $false
& $Py -c "import torch; print(torch.__version__)" | Out-Null
if ($LASTEXITCODE -eq 0) {
  if (Test-CudaOp) { $torchOk = $true }
}

if (-not $torchOk) {
  Write-Host "  - Installing Torch CUDA stable (thử trước)..."
  & $Py -m pip uninstall -y torch torchvision torchaudio | Out-Null

  $cudaIndexes = @(
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu121"
  )

  $installed = $false
  foreach ($idx in $cudaIndexes) {
    Write-Host "    * Torch index: $idx"
    & $Py -m pip install torch torchvision torchaudio --index-url $idx
    if ($LASTEXITCODE -eq 0) {
      $installed = $true
      break
    }
  }

  if (-not $installed) {
    throw "Không cài được Torch CUDA stable. Kiểm tra internet/driver." 
  }

  if (-not (Test-CudaOp)) {
    Write-Host "  ! Stable không chạy được CUDA kernels trên GPU này. Chuyển sang Torch nightly..."
    & $Py -m pip uninstall -y torch torchvision torchaudio | Out-Null

    $nightlyIndexes = @(
      "https://download.pytorch.org/whl/nightly/cu128",
      "https://download.pytorch.org/whl/nightly/cu126",
      "https://download.pytorch.org/whl/nightly/cu124"
    )

    $nightlyOk = $false
    foreach ($idx in $nightlyIndexes) {
      Write-Host "    * Nightly index: $idx"
      & $Py -m pip install --pre torch torchvision torchaudio --index-url $idx
      if (($LASTEXITCODE -eq 0) -and (Test-CudaOp)) {
        $nightlyOk = $true
        break
      }
    }

    if (-not $nightlyOk) {
      throw "Torch CUDA vẫn không chạy được CUDA kernels trên GPU này (ví dụ sm_120)."
    }
  }
}

& $Py -m pip install -U ultralytics opencv-python pyyaml | Out-Null

Write-Host "[2/3] Tạo pseudo-label dataset từ video..."
Push-Location $Repo

# Clean existing pseudo dataset to avoid accumulation
Remove-Item -Recurse -Force (Join-Path $Repo "datasets\pickleball\images") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $Repo "datasets\pickleball\labels") -ErrorAction SilentlyContinue

& $Py training/pseudolabel_from_videos.py --glob $Glob --sample-fps $SampleFps --val-ratio $ValRatio --max-frames-per-video $MaxFramesPerVideo --device $Device --imgsz $ImgSz --ball-conf $BallConf

Write-Host "[3/3] Train YOLO detect (pseudo-label)..."
& $Py training/train_detect.py --data training/pickleball.yaml --model $Model --name $Name --epochs $Epochs --batch $Batch --imgsz $ImgSz --device $Device

$bestPt = Get-ChildItem -Path (Join-Path $Repo "runs") -Recurse -Filter "best.pt" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if ($bestPt) {
  Write-Host ("Best weights (newest): " + $bestPt.FullName)
} else {
  Write-Host "[WARN] Không tìm thấy best.pt trong runs/." 
}

Write-Host "Done. Gợi ý dùng cho tracking: python yolo-tracking.py --det-model <duong_dan_best.pt>"
Pop-Location
