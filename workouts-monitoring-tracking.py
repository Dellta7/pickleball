from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import solutions


def process_video(
    video_path: Path,
    output_dir: Path,
    model_name: str,
    tracker: str,
    conf: float,
    iou: float,
    up_angle: float,
    down_angle: float,
    kpts: list[int],
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    output_path = output_dir / f"{video_path.stem}_workout_tracked.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    gym = solutions.AIGym(
        model=model_name,
        tracker=tracker,
        conf=conf,
        iou=iou,
        show=False,
        line_width=2,
        show_conf=True,
        show_labels=True,
        up_angle=up_angle,
        down_angle=down_angle,
        kpts=kpts,
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    print(f"\nProcessing: {video_path.name}")
    print(f"Resolution: {w}x{h} | FPS: {fps:.2f} | Frames: {total_frames}")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = gym(frame)
        writer.write(result.plot_im)

        frame_id += 1
        if total_frames > 0 and frame_id % max(1, total_frames // 10) == 0:
            print(f"  Progress: {100 * frame_id / total_frames:.1f}% ({frame_id}/{total_frames})")

    cap.release()
    writer.release()

    print(f"Done: {output_path}")
    return output_path


def parse_kpts(kpts_str: str) -> list[int]:
    parts = [x.strip() for x in kpts_str.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("kpts must contain exactly 3 indexes, e.g. 6,8,10")
    return [int(x) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ultralytics Workouts Monitoring (AIGym) with tracking and video export"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input-videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("workout-tracked-output"))
    parser.add_argument("--glob", type=str, default="*.mp4")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--up-angle", type=float, default=145.0)
    parser.add_argument("--down-angle", type=float, default=90.0)
    parser.add_argument(
        "--kpts",
        type=str,
        default="6,8,10",
        help="Three keypoint indexes. Example pushup: 6,8,10",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob(args.glob))
    if not videos:
        print(f"No videos found in {input_dir} with pattern {args.glob}")
        return

    kpts = parse_kpts(args.kpts)

    print("=" * 70)
    print("Workouts Monitoring + Tracking")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Videos: {len(videos)}")
    print(f"Model:  {args.model}")
    print(f"Tracker:{args.tracker}")
    print(f"Kpts:   {kpts}")
    print("=" * 70)

    done_count = 0
    skip_count = 0

    for video in videos:
        out_path = output_dir / f"{video.stem}_workout_tracked.mp4"
        if args.skip_existing and out_path.exists():
            print(f"Skip existing: {video.name}")
            skip_count += 1
            continue

        try:
            process_video(
                video_path=video,
                output_dir=output_dir,
                model_name=args.model,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                up_angle=args.up_angle,
                down_angle=args.down_angle,
                kpts=kpts,
            )
            done_count += 1
        except Exception as exc:
            print(f"Error: {video.name} -> {exc}")

    print("=" * 70)
    print(f"Completed: {done_count}/{len(videos)}")
    print(f"Skipped:   {skip_count}")
    print(f"Saved to:  {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
