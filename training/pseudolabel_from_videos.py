from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Expected output classes (MUST match classes.txt order)
# 0: pickleball_ball
# 1: player
# 2: paddle

# COCO ids (for default yolov8*.pt)
COCO_SPORTS_BALL = 32

# COCO-17 keypoints
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def resolve_ball_class_ids(det_model: YOLO) -> list[int] | None:
    names = getattr(det_model, "names", {}) or {}
    name_map = {int(k): str(v).strip().lower() for k, v in names.items()}
    ball_ids = [cid for cid, nm in name_map.items() if nm in {"sports ball", "pickleball_ball", "pickleball ball", "ball"}]
    if ball_ids:
        return sorted(set(ball_ids))
    if COCO_SPORTS_BALL in name_map:
        return [COCO_SPORTS_BALL]
    return None


def xyxy_to_yolo_line(cls_id: int, xyxy: tuple[float, float, float, float], w: int, h: int) -> str | None:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    # clamp
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def center_xyxy(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def area_xyxy(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def aspect_xyxy(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return float(w / h)


def get_wrist_points(
    kpts_xy: np.ndarray | None,
    kpts_conf: np.ndarray | None,
    conf_th: float = 0.25,
) -> list[tuple[float, float]]:
    if kpts_xy is None or kpts_xy.shape[0] < 11:
        return []
    pts: list[tuple[float, float]] = []
    for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
        if kpts_conf is None or float(kpts_conf[wi]) >= conf_th:
            pts.append((float(kpts_xy[wi, 0]), float(kpts_xy[wi, 1])))
    return pts


def pick_best_ball_box(
    ball_boxes: list[tuple[float, float, float, float]],
    person_box: tuple[float, float, float, float],
    wrist_pts: list[tuple[float, float]],
    last_center: tuple[float, float] | None,
) -> tuple[float, float, float, float] | None:
    if not ball_boxes:
        return None

    px1, py1, px2, py2 = [float(v) for v in person_box]
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    p_area = pw * ph
    p_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)

    # Ball is tiny; filter obvious false positives.
    filtered: list[tuple[float, float, float, float]] = []
    for b in ball_boxes:
        a = area_xyxy(b)
        if a <= 1.0:
            continue
        if a > 0.08 * p_area:
            continue
        asp = aspect_xyxy(b)
        if asp < 0.35 or asp > 2.85:
            continue
        bx1, by1, bx2, by2 = b
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        if bw > 0.35 * pw or bh > 0.35 * ph:
            continue
        filtered.append(b)

    candidates = filtered if filtered else ball_boxes

    target_pts = wrist_pts if wrist_pts else [p_center]
    scale = 0.35 * max(pw, ph)

    best = None
    best_cost = 1e18
    for b in candidates:
        bc = center_xyxy(b)
        d_hand = min(float(math.hypot(bc[0] - tx, bc[1] - ty)) for (tx, ty) in target_pts)
        d_last = float(math.hypot(bc[0] - last_center[0], bc[1] - last_center[1])) if last_center else 0.0
        # Prefer continuity if we have last center.
        cost = (d_hand / max(scale, 1.0)) + (0.55 * (d_last / max(scale, 1.0)) if last_center else 0.0)
        if cost < best_cost:
            best_cost = cost
            best = b

    return best


def estimate_paddle_box_from_pose(
    person_box_xyxy: tuple[float, float, float, float],
    kpts_xy: np.ndarray | None,
    kpts_conf: np.ndarray | None,
    ball_center: tuple[float, float] | None,
    conf_th: float = 0.25,
) -> tuple[float, float, float, float] | None:
    if kpts_xy is None or kpts_xy.shape[0] < 11:
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

    candidates: list[tuple[float, tuple[float, float]]] = []
    for wrist_i, elbow_i in ((KPT_LEFT_WRIST, KPT_LEFT_ELBOW), (KPT_RIGHT_WRIST, KPT_RIGHT_ELBOW)):
        if not (ok(wrist_i) and ok(elbow_i)):
            continue
        wx, wy = float(kpts_xy[wrist_i, 0]), float(kpts_xy[wrist_i, 1])
        ex, ey = float(kpts_xy[elbow_i, 0]), float(kpts_xy[elbow_i, 1])
        vx, vy = wx - ex, wy - ey
        n = float(math.hypot(vx, vy))
        if n < 1.0:
            continue
        vx, vy = vx / n, vy / n
        cx = wx + 0.35 * box_w * vx
        cy = wy + 0.35 * box_w * vy
        score = 0.0
        if ball_center is not None:
            score = -float(math.hypot(cx - ball_center[0], cy - ball_center[1]))
        candidates.append((score, (cx, cy)))

    if candidates:
        cx, cy = max(candidates, key=lambda t: t[0])[1]
    else:
        # fallback to any wrist
        wrists = []
        for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
            if ok(wi):
                wrists.append((float(kpts_xy[wi, 0]), float(kpts_xy[wi, 1])))
        if not wrists:
            return None
        cx, cy = wrists[0]

    return (cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0)


def pick_active_person_by_ball(
    person_xyxy: np.ndarray,
    person_kpts_xy: np.ndarray | None,
    person_kpts_conf: np.ndarray | None,
    ball_centers: list[tuple[float, float]],
) -> int | None:
    if person_xyxy.shape[0] == 0 or not ball_centers:
        return None

    best_i = None
    best_score = 1e9

    for i in range(person_xyxy.shape[0]):
        x1, y1, x2, y2 = [float(v) for v in person_xyxy[i]]
        pw = max(1.0, x2 - x1)
        ph = max(1.0, y2 - y1)
        scale = 0.35 * max(pw, ph)

        wrist_pts: list[tuple[float, float]] = []
        if person_kpts_xy is not None and i < person_kpts_xy.shape[0]:
            for wi in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
                if person_kpts_conf is None or float(person_kpts_conf[i, wi]) >= 0.25:
                    wrist_pts.append((float(person_kpts_xy[i, wi, 0]), float(person_kpts_xy[i, wi, 1])))

        if wrist_pts:
            dmin = 1e9
            for bc in ball_centers:
                for wpt in wrist_pts:
                    d = float(math.hypot(bc[0] - wpt[0], bc[1] - wpt[1]))
                    if d < dmin:
                        dmin = d
        else:
            pc = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            dmin = min(float(math.hypot(bc[0] - pc[0], bc[1] - pc[1])) for bc in ball_centers)

        score = dmin / max(scale, 1.0)
        if score < best_score:
            best_score = score
            best_i = i

    return best_i


def pick_largest_person(person_xyxy: np.ndarray) -> int | None:
    if person_xyxy.shape[0] == 0:
        return None
    areas = (person_xyxy[:, 2] - person_xyxy[:, 0]) * (person_xyxy[:, 3] - person_xyxy[:, 1])
    return int(np.argmax(areas))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pseudo-label pickleball YOLO detect dataset from videos (pose + ball + estimated paddle)",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input-videos"))
    parser.add_argument("--glob", type=str, default="*.mp4")
    parser.add_argument("--output", type=Path, default=Path("datasets/pickleball"))
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 = no limit")
    parser.add_argument("--pose-model", type=Path, default=Path("yolov8n-pose.pt"))
    parser.add_argument("--det-model", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--device", type=str, default="0", help="'0' for GPU, or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.05,
        help="Confidence for teacher ball detector; lower catches small balls but may add noise",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve() if not args.input_dir.is_absolute() else args.input_dir
    output = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output
    pose_model_path = (repo_root / args.pose_model).resolve() if not args.pose_model.is_absolute() else args.pose_model
    det_model_path = (repo_root / args.det_model).resolve() if not args.det_model.is_absolute() else args.det_model

    videos = sorted(input_dir.glob(args.glob))
    if not videos:
        raise SystemExit(f"No videos found in {input_dir} with pattern {args.glob}")

    rng = random.Random(args.seed)

    # Fresh output
    for p in [
        output / "images" / "train",
        output / "images" / "val",
        output / "labels" / "train",
        output / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    pose = YOLO(str(pose_model_path))
    det = YOLO(str(det_model_path))
    ball_class_ids = resolve_ball_class_ids(det)

    total_written = 0
    total_skipped = 0

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {video}")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if src_fps <= 0:
            src_fps = 30.0

        every = max(int(round(src_fps / max(args.sample_fps, 0.1))), 1)
        frame_idx = 0
        written_this_video = 0
        last_ball_center: tuple[float, float] | None = None

        print(f"Pseudo-label: {video.name} | src_fps={src_fps:.2f} | every={every} frames")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every != 0:
                frame_idx += 1
                continue

            if args.max_frames_per_video > 0 and written_this_video >= args.max_frames_per_video:
                break

            h, w = frame.shape[:2]

            # detect ball
            det_res = det.predict(
                frame,
                conf=float(args.ball_conf),
                iou=0.7,
                imgsz=args.imgsz,
                device=args.device,
                classes=ball_class_ids,
                verbose=False,
            )

            ball_boxes: list[tuple[float, float, float, float]] = []
            if det_res and det_res[0].boxes is not None and len(det_res[0].boxes) > 0:
                for b in det_res[0].boxes.xyxy.cpu().numpy():
                    ball_boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))

            ball_centers = [center_xyxy(b) for b in ball_boxes]

            # pose persons
            pose_res = pose.predict(
                frame,
                conf=0.25,
                iou=0.7,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            if not pose_res or pose_res[0].boxes is None or len(pose_res[0].boxes) == 0:
                total_skipped += 1
                frame_idx += 1
                continue

            person_xyxy = pose_res[0].boxes.xyxy.cpu().numpy()
            kpts_xy = pose_res[0].keypoints.xy.cpu().numpy() if pose_res[0].keypoints is not None else None
            kpts_conf = (
                pose_res[0].keypoints.conf.cpu().numpy()
                if pose_res[0].keypoints is not None and pose_res[0].keypoints.conf is not None
                else None
            )

            if ball_centers:
                active_i = pick_active_person_by_ball(person_xyxy, kpts_xy, kpts_conf, ball_centers)
            else:
                active_i = pick_largest_person(person_xyxy)

            if active_i is None:
                total_skipped += 1
                frame_idx += 1
                continue

            pbox = tuple(float(v) for v in person_xyxy[active_i])

            wrist_pts = get_wrist_points(
                kpts_xy[active_i] if kpts_xy is not None else None,
                kpts_conf[active_i] if kpts_conf is not None else None,
                conf_th=0.25,
            )

            # choose best ball (if any) using wrist proximity + temporal continuity
            best_ball = pick_best_ball_box(ball_boxes, pbox, wrist_pts, last_ball_center)
            bc = center_xyxy(best_ball) if best_ball is not None else None
            if bc is not None:
                last_ball_center = bc

            est_paddle = estimate_paddle_box_from_pose(
                pbox,
                kpts_xy[active_i] if kpts_xy is not None else None,
                kpts_conf[active_i] if kpts_conf is not None else None,
                bc,
            )

            # build YOLO label lines
            lines: list[str] = []
            l_ball = xyxy_to_yolo_line(0, best_ball, w, h) if best_ball is not None else None
            l_player = xyxy_to_yolo_line(1, pbox, w, h)
            if l_ball:
                lines.append(l_ball)
            if l_player:
                lines.append(l_player)
            if est_paddle is not None:
                l_pad = xyxy_to_yolo_line(2, est_paddle, w, h)
                if l_pad:
                    lines.append(l_pad)

            # Require at least player label
            if not l_player:
                total_skipped += 1
                frame_idx += 1
                continue

            # train/val split
            is_val = rng.random() < args.val_ratio
            split = "val" if is_val else "train"

            stem = f"{video.stem}_f{frame_idx:06d}"
            img_out = output / "images" / split / f"{stem}.jpg"
            lbl_out = output / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_out), frame)
            lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

            total_written += 1
            written_this_video += 1
            frame_idx += 1

        cap.release()

    print("Done pseudo-labeling")
    print(f"  output: {output}")
    print(f"  written: {total_written}")
    print(f"  skipped: {total_skipped}")


if __name__ == "__main__":
    main()
