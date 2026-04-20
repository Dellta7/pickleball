# Train YOLO cho Pickleball (detect/track)

Mục tiêu: train model detect riêng cho 3 lớp: `pickleball_ball`, `player`, `paddle`.

## Cách chạy tự động (khuyến nghị)

Nếu bạn có dataset dạng YOLO detect export (ví dụ Roboflow YOLOv8) dưới dạng folder hoặc `.zip`, bạn chỉ cần 1 lệnh:

- `powershell -ExecutionPolicy Bypass -File d:/Code/pickleball/training/auto_train.ps1 -Dataset "PATH_TO_DATASET_OR_ZIP" -Epochs 100 -Batch 8 -ImgSz 640 -Device 0`

Script sẽ tự:
- Cài Torch CUDA + Ultralytics trong venv
- Kiểm tra CUDA
- Import + chuẩn hoá dataset vào `datasets/pickleball/`
- Train và tạo `runs/detect/pickleball/weights/best.pt`

Ghi chú GPU mới (RTX 50xx / sm_120): nếu Torch stable chưa có kernel cho GPU của bạn, script sẽ tự fallback sang Torch nightly.

## Cách chạy tự động hoàn toàn (không cần gán nhãn thủ công)

Nếu bạn CHƯA có dataset bbox, bạn có thể tạo pseudo-label trực tiếp từ video (pose + bóng + ước lượng vợt), rồi train luôn:

- `powershell -ExecutionPolicy Bypass -File d:/Code/pickleball/training/auto_pseudo_and_train.ps1 -Glob "*.mp4" -SampleFps 6 -Epochs 60 -Batch 8 -ImgSz 640 -Device 0`

Tuỳ chọn hữu ích:
- `-BallConf 0.05` (hạ conf để bắt bóng nhỏ hơn, nhưng có thể nhiễu hơn)
- `-Model yolov8m.pt` (base model lớn hơn, thường học tốt hơn nếu GPU chịu được)
- `-Name pickleball_pseudo` (đặt tên run để dễ tìm output)

Script sẽ in ra đường dẫn `best.pt` mới nhất sau khi train.

Lưu ý: pseudo-label giúp khởi động nhanh, nhưng để accuracy cao bạn vẫn nên gán nhãn thủ công một phần và/hoặc sửa lại nhãn.

## 1) Chuẩn bị dataset (YOLO format)
Tạo dataset theo cấu trúc:

- `datasets/pickleball/images/train/*.jpg`
- `datasets/pickleball/images/val/*.jpg`
- `datasets/pickleball/labels/train/*.txt`
- `datasets/pickleball/labels/val/*.txt`

Mỗi file label `.txt` theo format YOLO:

`<class_id> <x_center> <y_center> <width> <height>`

Toạ độ là chuẩn hoá trong [0..1].

Class id phải đúng thứ tự trong `classes.txt`:
- `0`: `pickleball_ball`
- `1`: `player`
- `2`: `paddle`

## 2) Cập nhật đường dẫn YAML
Sửa `training/pickleball.yaml` cho đúng máy bạn:
- `path: d:/Code/pickleball/datasets/pickleball` (hoặc nơi bạn đặt dataset)

## 3) Train (GPU RTX 5060 8GB)
Chạy trong venv hiện tại:

- `d:/Code/pickleball/.venv/Scripts/python.exe d:/Code/pickleball/training/train_detect.py --epochs 100 --batch 8 --imgsz 640 --device 0`

Nếu bị OOM:
- giảm `--batch` xuống `4` hoặc `2`
- hoặc giảm `--imgsz` xuống `512`

Output sẽ nằm ở:
- `runs/detect/pickleball/weights/best.pt`

## 4) Dùng model đã train để tracking
Sau khi có `best.pt`, bạn có thể cập nhật `DET_MODEL_PATH` trong `yolo-tracking.py` trỏ tới file đó để detect đúng `paddle/ball`.

Hoặc chạy trực tiếp không cần sửa code:
- `d:/Code/pickleball/.venv/Scripts/python.exe d:/Code/pickleball/yolo-tracking.py --det-model runs/detect/pickleball/weights/best.pt --glob "ATP-shot.mp4"`

Gợi ý: dùng model custom detect + model pose `yolov8n-pose.pt` để có skeleton ổn định.
