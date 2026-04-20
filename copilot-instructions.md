# Copilot Instructions (Pickleball)

## Ngôn ngữ & phong cách
- Trả lời bằng tiếng Việt.
- Ưu tiên thay đổi tối thiểu, đúng trọng tâm.
- Khi sửa code Python: giữ style hiện có, không reformat lan man.

## Mục tiêu project
- Xử lý video pickleball: detect/track người chơi, vợt (paddle), bóng; overlay pose (người que) khi cần.
- Khi tạo output video tracking: chỉ ghi ra thư mục `tracked-output/` (không tạo thêm output phụ) trừ khi user yêu cầu rõ.

## Quy ước dữ liệu & lớp
- Danh sách lớp custom mong muốn: xem `classes.txt` (hiện: `pickleball_ball`, `player`, `paddle`).
- Khi train YOLO custom: dùng đúng thứ tự lớp theo `classes.txt`.

## Chạy/kiểm tra
- Nếu cần chạy nhanh: ưu tiên chạy 1 video bằng glob thay vì chạy cả thư mục.
- Trước khi kết luận “không hoạt động”: kiểm tra model đang dùng (COCO vs custom) và class mapping.

## Không làm
- Không tạo thêm folder output mới ngoài `tracked-output/` cho pipeline tracking.
- Không tự ý thêm UI/feature không được yêu cầu.
