from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def sanitize_name(name: str) -> str:
    """Keep filename safe across platforms while preserving readability."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def extract_video_frames(video_path: Path, output_root: Path, target_fps: float) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        src_fps = 30.0

    if target_fps <= 0:
        raise ValueError("target_fps phai lon hon 0")

    # Sample frame interval to get approximately target_fps frames each second.
    frame_interval = max(int(round(src_fps / target_fps)), 1)

    video_stem = sanitize_name(video_path.stem)
    video_output_dir = output_root / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    read_count = 0
    save_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if read_count % frame_interval == 0:
            frame_name = f"{video_stem}_frame_{save_count + 1:06d}.jpg"
            cv2.imwrite(str(video_output_dir / frame_name), frame)
            save_count += 1

        read_count += 1

    cap.release()
    actual_fps = src_fps / frame_interval
    return read_count, save_count, actual_fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Trich xuat frames de gan nhan YOLO")
    parser.add_argument("--input-dir", type=Path, default=Path("input-videos"), help="Thu muc chua video")
    parser.add_argument("--output-dir", type=Path, default=Path("frames"), help="Thu muc luu frame")
    parser.add_argument("--fps", type=float, default=8.0, help="So frame moi giay can lay (goi y 5-10)")
    parser.add_argument("--glob", type=str, default="*.mp4", help="Pattern tim video")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob(args.glob))
    if not videos:
        print(f"Khong tim thay video trong {input_dir} voi pattern {args.glob}")
        return

    print(f"Tim thay {len(videos)} video")
    print(f"Muc trich xuat muc tieu: {args.fps:.2f} fps")

    total_read = 0
    total_saved = 0

    for video in videos:
        try:
            read_count, save_count, actual_fps = extract_video_frames(video, output_dir, args.fps)
            total_read += read_count
            total_saved += save_count
            print(f"[OK] {video.name}: read={read_count}, saved={save_count}, actual_fps={actual_fps:.2f}")
        except Exception as exc:
            print(f"[ERROR] {video.name}: {exc}")

    print("-" * 60)
    print(f"Hoan tat. Tong read={total_read}, tong saved={total_saved}")
    print(f"Frames nam tai: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
