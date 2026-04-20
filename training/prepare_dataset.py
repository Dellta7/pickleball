from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Split:
    images_dir: Path
    labels_dir: Path


def read_classes_txt(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            names.append(s)
    return names


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def build_name_to_expected_id(expected_names: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for i, nm in enumerate(expected_names):
        mapping[normalize_name(nm)] = i

    # common synonyms
    if "pickleball ball" not in mapping and "pickleball ball" in mapping:
        pass

    # alias ball
    if normalize_name("pickleball_ball") in mapping:
        mapping.setdefault(normalize_name("ball"), mapping[normalize_name("pickleball_ball")])
        mapping.setdefault(normalize_name("pickleball ball"), mapping[normalize_name("pickleball_ball")])

    # alias paddle
    if normalize_name("paddle") in mapping:
        mapping.setdefault(normalize_name("racket"), mapping[normalize_name("paddle")])
        mapping.setdefault(normalize_name("tennis racket"), mapping[normalize_name("paddle")])

    # alias player
    if normalize_name("player") in mapping:
        mapping.setdefault(normalize_name("person"), mapping[normalize_name("player")])

    return mapping


def safe_unzip(zip_path: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

    # If zip contains a single top-level folder, return it.
    items = [p for p in dst_dir.iterdir()]
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return dst_dir


def find_data_yaml(root: Path) -> Path | None:
    for candidate in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = root / candidate
        if p.exists():
            return p

    # Search shallow
    for p in root.glob("**/data.yaml"):
        return p
    for p in root.glob("**/dataset.yaml"):
        return p
    return None


def resolve_split_from_yaml(dataset_root: Path, yaml_path: Path, key: str) -> Split | None:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return None

    base = obj.get("path", None)
    base_dir = Path(base) if isinstance(base, str) else dataset_root
    if not base_dir.is_absolute():
        base_dir = (yaml_path.parent / base_dir).resolve()

    val = obj.get(key)
    if not isinstance(val, str):
        return None

    split_images = (base_dir / val).resolve()

    # Common conventions
    # - images/train, labels/train
    # - train/images, train/labels
    parts = [p for p in split_images.parts]

    # If points directly to images directory
    if split_images.name == "images":
        split_root = split_images.parent
        images_dir = split_images
        labels_dir = split_root / "labels"
        return Split(images_dir=images_dir, labels_dir=labels_dir)

    # If path contains "images/<split>"
    try:
        idx = parts.index("images")
        split_root = Path(*parts[: idx + 1]).resolve()  # includes images
    except ValueError:
        split_root = None

    if split_images.exists():
        # best-effort infer labels
        if "images" in split_images.parts:
            # replace images with labels
            rel = split_images.relative_to(base_dir)
            rel_parts = list(rel.parts)
            rel_parts[rel_parts.index("images")] = "labels"
            labels_dir = (base_dir / Path(*rel_parts)).resolve()
            return Split(images_dir=split_images, labels_dir=labels_dir)

        # If points to split root like train/
        if (split_images / "images").exists():
            return Split(images_dir=(split_images / "images"), labels_dir=(split_images / "labels"))

    return None


def infer_splits(dataset_root: Path) -> tuple[Split | None, Split | None]:
    # Roboflow often: train/, valid/, test/
    train_root = None
    val_root = None

    for tr in ("train", "training"):
        p = dataset_root / tr
        if p.exists():
            train_root = p
            break

    for vr in ("valid", "val", "validation"):
        p = dataset_root / vr
        if p.exists():
            val_root = p
            break

    if train_root and (train_root / "images").exists():
        train = Split(images_dir=train_root / "images", labels_dir=train_root / "labels")
    elif (dataset_root / "images" / "train").exists():
        train = Split(images_dir=dataset_root / "images" / "train", labels_dir=dataset_root / "labels" / "train")
    else:
        train = None

    if val_root and (val_root / "images").exists():
        val = Split(images_dir=val_root / "images", labels_dir=val_root / "labels")
    elif (dataset_root / "images" / "val").exists():
        val = Split(images_dir=dataset_root / "images" / "val", labels_dir=dataset_root / "labels" / "val")
    else:
        val = None

    return train, val


def remap_label_file(src_label: Path, dst_label: Path, id_map: dict[int, int]) -> int:
    kept = 0
    lines_out: list[str] = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if cid not in id_map:
            continue
        parts[0] = str(id_map[cid])
        lines_out.append(" ".join(parts))
        kept += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept


def copy_split(
    split: Split,
    dst_images: Path,
    dst_labels: Path,
    id_map: dict[int, int],
) -> tuple[int, int]:
    if not split.images_dir.exists():
        raise SystemExit(f"Images dir not found: {split.images_dir}")
    if not split.labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {split.labels_dir}")

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    img_count = 0
    lbl_count = 0

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_files.extend(split.images_dir.glob(ext))

    for img in sorted(image_files):
        stem = img.stem
        src_lbl = split.labels_dir / f"{stem}.txt"
        if not src_lbl.exists():
            continue

        shutil.copy2(img, dst_images / img.name)
        kept = remap_label_file(src_lbl, dst_labels / src_lbl.name, id_map)
        if kept <= 0:
            # remove copied image/label if nothing kept
            try:
                (dst_images / img.name).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                (dst_labels / src_lbl.name).unlink(missing_ok=True)
            except Exception:
                pass
            continue

        img_count += 1
        lbl_count += 1

    return img_count, lbl_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare pickleball YOLO detect dataset (import zip/folder, normalize to datasets/pickleball)",
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source dataset folder or .zip (e.g., Roboflow YOLOv8 export)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("datasets/pickleball"),
        help="Destination dataset root (default: datasets/pickleball)",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("classes.txt"),
        help="Expected classes file (default: classes.txt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete dst folder before writing",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src = (repo_root / args.src).resolve() if not args.src.is_absolute() else args.src
    dst = (repo_root / args.dst).resolve() if not args.dst.is_absolute() else args.dst
    classes_path = (repo_root / args.classes).resolve() if not args.classes.is_absolute() else args.classes

    if not classes_path.exists():
        raise SystemExit(f"classes.txt not found: {classes_path}")

    expected_names = read_classes_txt(classes_path)
    expected_name_to_id = build_name_to_expected_id(expected_names)

    if args.force and dst.exists():
        shutil.rmtree(dst)

    extract_root = (repo_root / "datasets" / "_import").resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    if src.suffix.lower() == ".zip":
        extracted = safe_unzip(src, extract_root / src.stem)
        dataset_root = extracted
    else:
        dataset_root = src

    yaml_path = find_data_yaml(dataset_root)
    train_split = None
    val_split = None

    dataset_names: list[str] | None = None
    if yaml_path is not None:
        obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # names may be list or dict
            names = obj.get("names")
            if isinstance(names, list):
                dataset_names = [str(x) for x in names]
            elif isinstance(names, dict):
                # assume 0..n mapping
                dataset_names = [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]

        train_split = resolve_split_from_yaml(dataset_root, yaml_path, "train")
        val_split = resolve_split_from_yaml(dataset_root, yaml_path, "val") or resolve_split_from_yaml(dataset_root, yaml_path, "valid")

    if train_split is None or val_split is None:
        inf_train, inf_val = infer_splits(dataset_root)
        train_split = train_split or inf_train
        val_split = val_split or inf_val

    if train_split is None or val_split is None:
        raise SystemExit(
            "Cannot infer train/val splits. Provide a Roboflow-style export with data.yaml or a folder with train/valid splits."
        )

    if not dataset_names:
        # If not provided, assume source already uses expected ids
        id_map = {i: i for i in range(len(expected_names))}
    else:
        # Build id remap based on class name matching
        id_map: dict[int, int] = {}
        for src_id, src_name in enumerate(dataset_names):
            key = normalize_name(src_name)
            if key in expected_name_to_id:
                id_map[src_id] = expected_name_to_id[key]

        if not id_map:
            raise SystemExit(
                "Dataset class names do not match classes.txt. Please export YOLO with names matching: "
                + ", ".join(expected_names)
            )

    # Write normalized dataset
    img_tr, lbl_tr = copy_split(
        train_split,
        dst_images=dst / "images" / "train",
        dst_labels=dst / "labels" / "train",
        id_map=id_map,
    )
    img_val, lbl_val = copy_split(
        val_split,
        dst_images=dst / "images" / "val",
        dst_labels=dst / "labels" / "val",
        id_map=id_map,
    )

    print("Prepared dataset:")
    print(f"  dst: {dst}")
    print(f"  train: images={img_tr}, labels={lbl_tr}")
    print(f"  val:   images={img_val}, labels={lbl_val}")


if __name__ == "__main__":
    main()
