# Datasets

Đây là nơi bạn đặt dataset để train YOLO.

## Cách nhanh nhất
Nếu bạn có dataset YOLO detect export (ví dụ Roboflow YOLOv8) dưới dạng folder hoặc `.zip`, bạn có thể để ở bất kỳ đâu rồi chạy:

- `powershell -ExecutionPolicy Bypass -File d:/Code/pickleball/training/auto_train.ps1 -Dataset "PATH_TO_DATASET_OR_ZIP"`

Script sẽ tự import + chuẩn hoá vào `datasets/pickleball/`.

## Dataset sau khi chuẩn hoá
Pipeline sẽ tạo:
- `datasets/pickleball/images/train`
- `datasets/pickleball/images/val`
- `datasets/pickleball/labels/train`
- `datasets/pickleball/labels/val`

Class order theo `classes.txt`:
- 0: pickleball_ball
- 1: player
- 2: paddle
