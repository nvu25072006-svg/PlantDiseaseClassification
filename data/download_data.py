import argparse
import shutil
import zipfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR


def import_local_plantvillage(source_path: Path, target_dir: Path, max_per_class: int = 0) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    extract_root = target_dir.parent / "PlantVillage_extracted"
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(extract_root)
        search_root = extract_root
    else:
        search_root = source_path

    candidate_dirs = list(search_root.rglob("color")) + list(search_root.rglob("PlantVillage"))
    source_class_root = None
    for cand in candidate_dirs:
        if cand.is_dir() and any(d.is_dir() for d in cand.iterdir()):
            source_class_root = cand
            break
    if source_class_root is None:
        source_class_root = search_root

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    total_copied = 0
    class_count = 0

    for cls_dir in sorted(source_class_root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        imgs = [f for f in sorted(cls_dir.iterdir()) if f.suffix in valid_exts]
        if not imgs:
            continue
        class_count += 1
        if max_per_class > 0:
            imgs = imgs[:max_per_class]
        dest_cls = target_dir / cls_dir.name
        dest_cls.mkdir(parents=True, exist_ok=True)
        for img_file in imgs:
            shutil.copy2(img_file, dest_cls / img_file.name)
            total_copied += 1

    print(f"Loaded {total_copied} images across {class_count} classes into {target_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_path",
        type=str,
        default=str(Path.home() / "Downloads" / "PlantVillage-Dataset-master.zip"),
    )
    parser.add_argument("--max_per_class", type=int, default=0)
    args = parser.parse_args()
    import_local_plantvillage(Path(args.local_path), DATA_DIR, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
