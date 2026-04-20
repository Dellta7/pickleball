Param(
  [Parameter(Mandatory=$true)]
  [string]$Dataset,

  [int]$Epochs = 100,
  [int]$Batch = 8,
  [int]$ImgSz = 640,
  [string]$Device = "0"
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Repo  # go up from training/

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  throw "Không thấy venv tại $Py. Hãy tạo venv trước hoặc dùng đúng workspace." 
}

function Test-CudaOp {
  & $Py -c "import torch; x=torch.randn((256,256), device='cuda'); y=x@x; print('cuda_ok')" | Out-Null
  return ($LASTEXITCODE -eq 0)
}

Write-Host "[1/4] Đảm bảo dependencies (Torch CUDA + Ultralytics)..."
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

& $Py -m pip install -U ultralytics opencv-python pyyaml

Write-Host "[2/4] Kiểm tra CUDA..."
& $Py -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"

Write-Host "[3/4] Chuẩn hoá dataset YOLO..."
Push-Location $Repo
& $Py training/prepare_dataset.py --src $Dataset --dst datasets/pickleball --classes classes.txt --force

Write-Host "[4/4] Train YOLO detect..."
& $Py training/train_detect.py --data training/pickleball.yaml --epochs $Epochs --batch $Batch --imgsz $ImgSz --device $Device

Write-Host "Done. Best checkpoint thường nằm ở runs/detect/pickleball/weights/best.pt"
Write-Host "Ví dụ chạy tracking:"
Write-Host "  $Py yolo-tracking.py --det-model runs/detect/pickleball/weights/best.pt --glob \"ATP-shot.mp4\""
Pop-Location
