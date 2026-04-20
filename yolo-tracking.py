"""YOLO tracking cho Pickleball.

Yêu cầu (theo mô tả của bạn):
- Chỉ tracking người cầm vợt (ưu tiên người đang tương tác với bóng).
- Người được track hiển thị dạng người que (pose skeleton) nhưng vẫn giữ:
  ô track người, vợt, chuyển động (trajectory) và bóng.
- Chỉ xuất video vào thư mục tracked-output (không tạo output phụ).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Cấu hình thư mục
VIDEO_DIR = Path(__file__).resolve().parent
INPUT_DIR = VIDEO_DIR / "input-videos"
OUTPUT_DIR = VIDEO_DIR / "tracked-output"
OUTPUT_DIR.mkdir(exist_ok=True)


# Model paths (dùng file local trong workspace để tránh auto-download)
DET_MODEL_PATH = VIDEO_DIR / "yolov8n.pt"
POSE_MODEL_PATH = VIDEO_DIR / "yolov8n-pose.pt"


# COCO class ids (YOLOv8n default)
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
COCO_TENNIS_RACKET = 38


# COCO-17 keypoint indexes (YOLOv8 pose)
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8


SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _as_int_tuple(xy: tuple[float, float]) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def xywh_to_xyxy(xywh: np.ndarray) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in xywh]
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def center_xyxy(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def draw_box(im: np.ndarray, xyxy: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(im.shape[1] - 1, x2)
    y2 = min(im.shape[0] - 1, y2)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(im, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def draw_trajectory(im: np.ndarray, pts: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(im, [arr], isClosed=False, color=color, thickness=2)


def draw_skeleton(
    im: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    color: tuple[int, int, int] = (255, 255, 255),
    kpt_radius: int = 3,
    conf_th: float = 0.25,
) -> None:
    if kpts_xy is None or len(kpts_xy) == 0:
        return

    def ok(i: int) -> bool:
        if kpts_conf is None:
            return True
        return float(kpts_conf[i]) >= conf_th

    # Edges
    for a, b in SKELETON_EDGES:
        if ok(a) and ok(b):
            pa = _as_int_tuple((kpts_xy[a, 0], kpts_xy[a, 1]))
            pb = _as_int_tuple((kpts_xy[b, 0], kpts_xy[b, 1]))
            cv2.line(im, pa, pb, color, 2, cv2.LINE_AA)

    # Keypoints
    for i in range(kpts_xy.shape[0]):
        if ok(i):
            p = _as_int_tuple((kpts_xy[i, 0], kpts_xy[i, 1]))
            cv2.circle(im, p, kpt_radius, color, -1, cv2.LINE_AA)


def estimate_paddle_box_from_pose(
    person_box_xyxy: tuple[float, float, float, float],
    kpts_xy: np.ndarray | None,
    kpts_conf: np.ndarray | None,
    ball_center: tuple[float, float] | None,
    conf_th: float = 0.25,
) -> tuple[float, float, float, float] | None:
    """Ước lượng bbox vợt/paddle quanh cổ tay nếu không detect được object vợt.

    Dùng hướng cẳng tay (elbow->wrist) để đẩy điểm center ra ngoài một chút.
    """
    if kpts_xy is None or len(kpts_xy) < 11:
        return None

    x1, y1, x2, y2 = [float(v) for v in person_box_xyxy]
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    box_w = 0.18 * pw
    box_h = 0.10 * ph

    def ok(i: int) -> bool:
        if kpts_conf is None:
            return True
        return float(kpts_conf[i]) >= conf_th

    candidates = []
    for wrist_i, elbow_i in ((KPT_LEFT_WRIST, KPT_LEFT_ELBOW), (KPT_RIGHT_WRIST, KPT_RIGHT_ELBOW)):
        if not (ok(wrist_i) and ok(elbow_i)):
            continue

        wx, wy = float(kpts_xy[wrist_i, 0]), float(kpts_xy[wrist_i, 1])
        ex, ey = float(kpts_xy[elbow_i, 0]), float(kpts_xy[elbow_i, 1])
        vx, vy = wx - ex, wy - ey
        n = float(np.hypot(vx, vy))
        if n < 1.0:
            continue
        vx, vy = vx / n, vy / n

        # đẩy ra ngoài theo hướng cẳng tay
        cx = wx + 0.35 * box_w * vx
        cy = wy + 0.35 * box_w * vy

        score = 0.0
        if ball_center is not None:
            score = -float(np.hypot(cx - ball_center[0], cy - ball_center[1]))
        candidates.append((score, (cx, cy)))

    if not candidates:
        # fallback: dùng wrist có conf cao (nếu có)
        wrists = []
        for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
            if ok(wi):
                wrists.append((float(kpts_xy[wi, 0]), float(kpts_xy[wi, 1])))
        if not wrists:
            return None
        cx, cy = wrists[0]
    else:
        cx, cy = max(candidates, key=lambda t: t[0])[1]

    return (cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0)


def pick_active_person(
    person_ids: list[int],
    person_xyxy: np.ndarray,
    person_kpts_xy: np.ndarray | None,
    person_kpts_conf: np.ndarray | None,
    racket_xyxy: list[tuple[float, float, float, float]],
    ball_xyxy: list[tuple[float, float, float, float]],
    prev_active_id: int | None,
) -> tuple[int | None, int | None]:
    """Chọn 1 người cầm vợt.

    Trả về (active_person_id, active_person_index_in_arrays).
    """
    if not person_ids:
        return None, None

    racket_centers = [center_xyxy(b) for b in racket_xyxy]
    ball_centers = [center_xyxy(b) for b in ball_xyxy]

    best_idx: int | None = None
    best_score = -1e9

    for i, pid in enumerate(person_ids):
        x1, y1, x2, y2 = [float(v) for v in person_xyxy[i]]
        pw = max(1.0, x2 - x1)
        ph = max(1.0, y2 - y1)
        scale = 0.35 * max(pw, ph)
        person_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # 1) Hold score: vợt gần cổ tay (nếu detect được vợt)
        hold_score = 0.0
        best_racket_dist = None

        if racket_centers and person_kpts_xy is not None and i < person_kpts_xy.shape[0]:
            wrists: list[tuple[float, float]] = []
            for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
                if person_kpts_conf is None or float(person_kpts_conf[i, wi]) >= 0.25:
                    wrists.append((float(person_kpts_xy[i, wi, 0]), float(person_kpts_xy[i, wi, 1])))

            if wrists:
                dmin = 1e9
                for rc in racket_centers:
                    for wpt in wrists:
                        d = float(np.hypot(rc[0] - wpt[0], rc[1] - wpt[1]))
                        if d < dmin:
                            dmin = d
                best_racket_dist = dmin
        elif racket_centers:
            # fallback: dùng khoảng cách đến tâm người
            dmin = min(float(np.hypot(rc[0] - person_center[0], rc[1] - person_center[1])) for rc in racket_centers)
            best_racket_dist = dmin

        if best_racket_dist is not None:
            # score ~ 1 khi rất gần cổ tay, giảm dần theo khoảng cách
            hold_score = max(0.0, 1.0 - (best_racket_dist / max(scale, 1.0)))

        # 2) Ball score: bóng gần cổ tay (quan trọng khi không detect được vợt)
        ball_score = 0.0
        best_ball_dist = None
        if ball_centers:
            # nếu có wrist thì dùng wrist, không thì dùng tâm người
            wrist_pts: list[tuple[float, float]] = []
            if person_kpts_xy is not None and i < person_kpts_xy.shape[0]:
                for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
                    if person_kpts_conf is None or float(person_kpts_conf[i, wi]) >= 0.25:
                        wrist_pts.append((float(person_kpts_xy[i, wi, 0]), float(person_kpts_xy[i, wi, 1])))

            if wrist_pts:
                dmin = 1e9
                for bc in ball_centers:
                    for wpt in wrist_pts:
                        d = float(np.hypot(bc[0] - wpt[0], bc[1] - wpt[1]))
                        if d < dmin:
                            dmin = d
                best_ball_dist = dmin
            else:
                best_ball_dist = min(float(np.hypot(bc[0] - person_center[0], bc[1] - person_center[1])) for bc in ball_centers)

        if best_ball_dist is not None:
            ball_score = max(0.0, 1.0 - (best_ball_dist / max(scale, 1.0)))

        # 3) Stability: ưu tiên giữ người đã chọn từ frame trước
        stability = 0.15 if (prev_active_id is not None and pid == prev_active_id) else 0.0

        # Nếu không detect được vợt, dựa chủ yếu vào bóng gần cổ tay.
        if racket_centers:
            score = 1.0 * hold_score + 0.45 * ball_score + stability
        else:
            score = 1.15 * ball_score + stability

        if score > best_score:
            best_score = score
            best_idx = i

    # Nếu không ai đủ "active" rõ ràng: chỉ giữ người cũ (nếu còn), còn lại không chọn ai.
    if best_idx is None or best_score < 0.10:
        if prev_active_id is not None and prev_active_id in person_ids:
            return int(prev_active_id), int(person_ids.index(prev_active_id))
        return None, None

    return int(person_ids[best_idx]), int(best_idx)


def resolve_det_class_ids(det_model: YOLO) -> tuple[set[int], set[int]]:
    """Resolve class IDs for (ball, paddle/racket) from model.names.

    Supports:
    - COCO: 'sports ball', 'tennis racket'
    - Custom: 'pickleball_ball', 'paddle'
    """
    names = getattr(det_model, "names", {}) or {}
    name_map = {int(k): str(v).strip().lower() for k, v in names.items()}

    ball_ids: set[int] = set()
    paddle_ids: set[int] = set()

    for cid, nm in name_map.items():
        if nm in {"sports ball", "pickleball_ball", "pickleball ball"} or nm.endswith("_ball") or nm == "ball":
            ball_ids.add(cid)
        if nm in {"tennis racket", "paddle", "pickleball_paddle", "pickleball paddle"}:
            paddle_ids.add(cid)
        # heuristic fallback
        if "racket" in nm and cid not in paddle_ids:
            paddle_ids.add(cid)

    # COCO fallback if not found by name
    if not ball_ids and COCO_SPORTS_BALL in name_map:
        ball_ids.add(COCO_SPORTS_BALL)
    if not paddle_ids and COCO_TENNIS_RACKET in name_map:
        paddle_ids.add(COCO_TENNIS_RACKET)

    return ball_ids, paddle_ids

def get_video_list(input_dir: Path, pattern: str):
    return sorted(input_dir.glob(pattern))


def process_video_with_trajectories(
    video_path,
    output_dir: Path,
    det_model_path: Path,
    pose_model_path: Path,
):
    """
    Xử lý video với YOLO tracking và hiển thị quỹ đạo của các objects
    Sử dụng persist=True để giữ track ID nhất quán giữa các frames
    """
    print(f"\n{'='*70}")
    print(f"🎬 Xử lý video: {video_path.name}")
    print(f"{'='*70}")
    
    # Kiểm tra file tồn tại
    if not video_path.exists():
        print(f"❌ File không tồn tại: {video_path}")
        return
    
    # Mở video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return
    
    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Thông tin video:")
    print(f"   • FPS: {fps}")
    print(f"   • Độ phân giải: {width}x{height}")
    print(f"   • Tổng frames: {total_frames}")
    
    # Thiết lập output video
    output_path = output_dir / f"{video_path.stem}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Load models (global) - lazy load để in thông báo 1 lần
    global det_model, pose_model  # noqa: PLW0603
    if "det_model" not in globals() or str(getattr(det_model, "ckpt_path", "")) != str(det_model_path):
        print("Đang tải model detect (ball/racket)...")
        det_model = YOLO(str(det_model_path))
        det_model.ckpt_path = str(det_model_path)
    if "pose_model" not in globals() or str(getattr(pose_model, "ckpt_path", "")) != str(pose_model_path):
        print("Đang tải model pose (skeleton)...")
        pose_model = YOLO(str(pose_model_path))
        pose_model.ckpt_path = str(pose_model_path)

    ball_ids, paddle_ids = resolve_det_class_ids(det_model)
    det_classes = sorted(ball_ids | paddle_ids) if (ball_ids or paddle_ids) else None

    # Lưu lịch sử quỹ đạo (chỉ cho player/paddle/ball)
    track_history: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)

    # Trạng thái người đang được chọn
    active_person_id: int | None = None
    active_person_miss = 0
    
    print(f"\n🎯 Bắt đầu tracking...")
    frame_count = 0
    # Thống kê
    tracked_objects = {
        "player_ids": set(),
        "paddle_ids": set(),
        "ball_ids": set(),
    }
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # 1) Pose tracking (người)
        pose_res = pose_model.track(
            frame,
            persist=True,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )

        person_ids: list[int] = []
        person_xyxy = np.zeros((0, 4), dtype=np.float32)
        person_kpts_xy = None
        person_kpts_conf = None

        if pose_res and pose_res[0].boxes is not None:
            person_xyxy = pose_res[0].boxes.xyxy.cpu().numpy()
            if pose_res[0].boxes.id is not None:
                person_ids = pose_res[0].boxes.id.int().cpu().tolist()
            else:
                # Fallback: tạo ID tạm theo index (không ổn định giữa frames, nhưng tránh mất render)
                person_ids = list(range(1, len(person_xyxy) + 1))

            if pose_res[0].keypoints is not None:
                person_kpts_xy = pose_res[0].keypoints.xy.cpu().numpy()
                if pose_res[0].keypoints.conf is not None:
                    person_kpts_conf = pose_res[0].keypoints.conf.cpu().numpy()

        # 2) Object tracking (ball + racket)
        det_res = det_model.track(
            frame,
            persist=True,
            conf=0.15,
            iou=0.7,
            classes=det_classes,
            verbose=False,
        )

        racket_items: list[tuple[int, float, tuple[float, float, float, float]]] = []
        ball_items: list[tuple[int, float, tuple[float, float, float, float]]] = []

        if det_res and det_res[0].boxes is not None:
            det_xywh = det_res[0].boxes.xywh.cpu().numpy()
            if det_res[0].boxes.id is not None:
                det_ids = det_res[0].boxes.id.int().cpu().tolist()
            else:
                det_ids = list(range(1, len(det_xywh) + 1))
            det_cls = det_res[0].boxes.cls.int().cpu().tolist()
            det_conf = det_res[0].boxes.conf.cpu().tolist()

            for xywh, tid, cls, conf in zip(det_xywh, det_ids, det_cls, det_conf):
                xyxy = xywh_to_xyxy(xywh)
                if int(cls) in paddle_ids:
                    racket_items.append((int(tid), float(conf), xyxy))
                elif int(cls) in ball_ids:
                    ball_items.append((int(tid), float(conf), xyxy))

        racket_xyxy = [b for _, _, b in racket_items]
        ball_xyxy = [b for _, _, b in ball_items]

        # 3) Chọn đúng người cần tracking
        picked_id, picked_idx = pick_active_person(
            person_ids=person_ids,
            person_xyxy=person_xyxy,
            person_kpts_xy=person_kpts_xy,
            person_kpts_conf=person_kpts_conf,
            racket_xyxy=racket_xyxy,
            ball_xyxy=ball_xyxy,
            prev_active_id=active_person_id,
        )

        # Giữ ổn định: nếu mất người vài frame thì mới reset
        if picked_id is None:
            active_person_miss += 1
            if active_person_miss >= 10:
                active_person_id = None
                active_person_miss = 0
        else:
            active_person_id = picked_id
            active_person_miss = 0

        # Khôi phục active_idx theo track id để render không bị "rớt" ở các frame thiếu vợt/bóng.
        active_idx = picked_idx
        if active_person_id is not None and active_idx is None:
            try:
                active_idx = person_ids.index(int(active_person_id))
            except ValueError:
                active_idx = None

        annotated_frame = frame.copy()

        # 4) Render: chỉ vẽ người được chọn + vợt + bóng
        # Người
        if active_person_id is not None and active_idx is not None:
            pid = int(active_person_id)
            pbox = tuple(float(v) for v in person_xyxy[active_idx])
            draw_box(annotated_frame, pbox, (255, 80, 80), f"player #{pid}")

            if person_kpts_xy is not None and active_idx < person_kpts_xy.shape[0]:
                kxy = person_kpts_xy[active_idx]
                kcf = person_kpts_conf[active_idx] if person_kpts_conf is not None else None
                draw_skeleton(annotated_frame, kxy, kcf, color=(255, 255, 255))

            tracked_objects["player_ids"].add(pid)
            pc = center_xyxy(pbox)
            track_history[("player", pid)].append(pc)
            track_history[("player", pid)] = track_history[("player", pid)][-50:]
            draw_trajectory(annotated_frame, track_history[("player", pid)], (0, 255, 0))

            # Chọn vợt gần cổ tay nhất của người này
            chosen_racket: tuple[int, float, tuple[float, float, float, float]] | None = None
            if racket_items:
                wrists = []
                if person_kpts_xy is not None:
                    for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
                        if person_kpts_conf is None or float(person_kpts_conf[active_idx, wi]) >= 0.25:
                            wrists.append(
                                (float(person_kpts_xy[active_idx, wi, 0]), float(person_kpts_xy[active_idx, wi, 1]))
                            )

                best_d = 1e9
                for rid, rconf, rbox in racket_items:
                    rc = center_xyxy(rbox)
                    if wrists:
                        d = min(float(np.hypot(rc[0] - w[0], rc[1] - w[1])) for w in wrists)
                    else:
                        d = float(np.hypot(rc[0] - pc[0], rc[1] - pc[1]))
                    if d < best_d:
                        best_d = d
                        chosen_racket = (rid, rconf, rbox)

            # Vẽ vợt
            racket_center = None
            if chosen_racket is not None:
                rid, rconf, rbox = chosen_racket
                draw_box(annotated_frame, rbox, (80, 255, 255), f"paddle #{rid}")
                tracked_objects["paddle_ids"].add(int(rid))
                racket_center = center_xyxy(rbox)
                track_history[("paddle", int(rid))].append(racket_center)
                track_history[("paddle", int(rid))] = track_history[("paddle", int(rid))][-50:]
                draw_trajectory(annotated_frame, track_history[("paddle", int(rid))], (80, 255, 255))
            else:
                # Không detect được vợt -> ước lượng bbox quanh cổ tay
                kxy_full = person_kpts_xy[active_idx] if person_kpts_xy is not None else None
                kcf_full = person_kpts_conf[active_idx] if person_kpts_conf is not None else None
                # ball center (nếu có) để chọn tay phù hợp
                ball_center_hint = None
                if ball_items:
                    # lấy bóng gần người nhất làm hint
                    best = None
                    for _, _, bbox in ball_items:
                        bc = center_xyxy(bbox)
                        d = float(np.hypot(bc[0] - pc[0], bc[1] - pc[1]))
                        if best is None or d < best[0]:
                            best = (d, bc)
                    if best is not None:
                        ball_center_hint = best[1]

                est = estimate_paddle_box_from_pose(pbox, kxy_full, kcf_full, ball_center_hint)
                if est is not None:
                    # dùng pid như track id cho paddle ước lượng (ổn định theo người)
                    rid = pid
                    draw_box(annotated_frame, est, (80, 255, 255), f"paddle #{rid}")
                    tracked_objects["paddle_ids"].add(int(rid))
                    racket_center = center_xyxy(est)
                    track_history[("paddle", int(rid))].append(racket_center)
                    track_history[("paddle", int(rid))] = track_history[("paddle", int(rid))][-50:]
                    draw_trajectory(annotated_frame, track_history[("paddle", int(rid))], (80, 255, 255))

            # Chọn bóng gần vợt nhất (nếu có vợt), fallback gần người
            chosen_ball: tuple[int, float, tuple[float, float, float, float]] | None = None
            if ball_items:
                target = racket_center if racket_center is not None else pc
                best_bd = 1e9
                for bid, bconf, bbox in ball_items:
                    bc = center_xyxy(bbox)
                    d = float(np.hypot(bc[0] - target[0], bc[1] - target[1]))
                    if d < best_bd:
                        best_bd = d
                        chosen_ball = (bid, bconf, bbox)

            if chosen_ball is not None:
                bid, bconf, bbox = chosen_ball
                draw_box(annotated_frame, bbox, (80, 80, 255), f"ball #{bid}")
                tracked_objects["ball_ids"].add(int(bid))
                bc = center_xyxy(bbox)
                track_history[("ball", int(bid))].append(bc)
                track_history[("ball", int(bid))] = track_history[("ball", int(bid))][-50:]
                draw_trajectory(annotated_frame, track_history[("ball", int(bid))], (80, 80, 255))
        
        # Ghi frame vào output video
        out.write(annotated_frame)
        
        # Hiển thị tiến độ
        if frame_count % max(1, total_frames // 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   ✓ Tiến độ: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    
    print(f"\n✅ Hoàn thành xử lý video")
    print(f"📁 Output: {output_path}")
    print(f"\n📈 Thống kê tracking:")
    print(f"   • Player IDs: {len(tracked_objects['player_ids'])}")
    print(f"   • Paddle IDs: {len(tracked_objects['paddle_ids'])}")
    print(f"   • Ball IDs:   {len(tracked_objects['ball_ids'])}")
    
    return output_path


def tracked_output_path(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}_tracked.mp4"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Track objects and export tracked videos")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Directory containing source videos")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for tracked videos")
    parser.add_argument(
        "--det-model",
        type=Path,
        default=DET_MODEL_PATH,
        help="Detect model (.pt). Use your trained best.pt for ball/paddle.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=POSE_MODEL_PATH,
        help="Pose model (.pt). Default yolov8n-pose.pt.",
    )
    parser.add_argument("--glob", type=str, default="*.mp4", help="Video pattern, e.g. *.mp4")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos that already have tracked output")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = get_video_list(args.input_dir, args.glob)
    if not videos:
        print(f"❌ Không tìm thấy video trong: {args.input_dir} (pattern={args.glob})")
        return

    print("\n" + "="*70)
    print("🎯 YOLO TRACKING CHO PICKLEBALL VIDEOS (Official Documentation)")
    print("="*70)
    print("📌 Sử dụng: persist=True, trajectory plotting, optimal settings\n")
    print(f"📂 Input: {args.input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"🎞️ Tổng video cần xử lý: {len(videos)}\n")
    
    processed_count = 0
    skipped_count = 0
    
    for video_path in videos:
        try:
            if args.skip_existing and tracked_output_path(video_path, output_dir).exists():
                print(f"⏭️ Bỏ qua (đã có output): {video_path.name}")
                skipped_count += 1
                continue

            output = process_video_with_trajectories(
                video_path,
                output_dir,
                det_model_path=args.det_model,
                pose_model_path=args.pose_model,
            )
            if output:
                processed_count += 1
        except Exception as e:
            print(f"❌ Lỗi xử lý {video_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ Hoàn thành! Đã xử lý {processed_count}/{len(videos)} video")
    print(f"⏭️ Đã bỏ qua: {skipped_count} video")
    print(f"📁 Output folder: {output_dir}")
    print(f"🎬 Video với trajectories được lưu trong thư mục trên")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
