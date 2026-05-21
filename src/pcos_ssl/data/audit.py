from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_TO_LABEL = {"infected": 1, "noninfected": 0}


@dataclass
class ImageRecord:
    image_id: str
    rel_path: str
    filename: str
    class_name: str
    label: int
    suffix: str
    bytes: int
    readable: bool
    width: int | None
    height: int | None
    mode: str | None
    md5: str | None
    phash: str | None
    error: str | None


def iter_image_paths(dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for class_name in sorted(CLASS_TO_LABEL):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(path)
    return paths


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_one(path: Path, dataset_root: Path) -> ImageRecord:
    class_name = path.parent.name
    rel_path = path.relative_to(dataset_root.parent).as_posix()
    image_id = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]

    try:
        file_hash = md5_file(path)
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            perceptual_hash = str(imagehash.phash(image))
        return ImageRecord(
            image_id=image_id,
            rel_path=rel_path,
            filename=path.name,
            class_name=class_name,
            label=CLASS_TO_LABEL[class_name],
            suffix=path.suffix.lower(),
            bytes=path.stat().st_size,
            readable=True,
            width=width,
            height=height,
            mode=mode,
            md5=file_hash,
            phash=perceptual_hash,
            error=None,
        )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return ImageRecord(
            image_id=image_id,
            rel_path=rel_path,
            filename=path.name,
            class_name=class_name,
            label=CLASS_TO_LABEL.get(class_name, -1),
            suffix=path.suffix.lower(),
            bytes=path.stat().st_size if path.exists() else 0,
            readable=False,
            width=None,
            height=None,
            mode=None,
            md5=None,
            phash=None,
            error=repr(exc),
        )


def audit_dataset(dataset_root: Path, workers: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = iter_image_paths(dataset_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            tqdm(
                pool.map(lambda p: audit_one(p, dataset_root), paths),
                total=len(paths),
                desc="Auditing images",
            )
        )

    images = pd.DataFrame([asdict(record) for record in records])
    if images.empty:
        return images, pd.DataFrame()

    images["exact_group_id"] = images["md5"].map(
        lambda value: f"md5_{value}" if isinstance(value, str) else None
    )

    readable = images[images["readable"] & images["md5"].notna()].copy()
    grouped = (
        readable.groupby(["exact_group_id", "md5", "class_name", "label"], dropna=False)
        .agg(
            group_size=("image_id", "size"),
            representative_rel_path=("rel_path", "first"),
            total_bytes=("bytes", "sum"),
        )
        .reset_index()
        .sort_values(["group_size", "class_name"], ascending=[False, True])
    )
    duplicate_groups = grouped[grouped["group_size"] > 1].reset_index(drop=True)

    return images.sort_values("rel_path").reset_index(drop=True), duplicate_groups


def write_audit(dataset_root: Path, output_dir: Path, workers: int = 18) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images, duplicate_groups = audit_dataset(dataset_root=dataset_root, workers=workers)

    images_path = output_dir / "images.csv"
    duplicate_groups_path = output_dir / "duplicate_groups.csv"
    images.to_csv(images_path, index=False)
    duplicate_groups.to_csv(duplicate_groups_path, index=False)
    return images_path, duplicate_groups_path
