from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from rich.console import Console


CANVAS = "white"
TEXT = (30, 30, 30)
MUTED = (95, 95, 95)
BORDER = (190, 190, 190)
PCOS = (160, 50, 45)
HEALTHY = (35, 95, 155)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript image example panels.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, text, fill=fill)


def load_square(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (248, 248, 248))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def add_tile(
    canvas: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    draw.rectangle((x - 2, y - 2, x + image.width + 1, y + image.height + 1), outline=BORDER, width=2)
    canvas.paste(image, (x, y))
    draw.rectangle((x, y + image.height - 28, x + image.width, y + image.height), fill=(255, 255, 255))
    draw_label(draw, (x + 8, y + image.height - 24), title, accent)
    draw_label(draw, (x + 8, y + image.height + 8), subtitle, MUTED)


def build_dataset_panel(project_root: Path, output_dir: Path) -> Path:
    images = pd.read_csv(project_root / "metadata/images.csv")
    groups = pd.read_csv(project_root / "metadata/near_duplicate_groups_phash.csv")
    merged = images.merge(
        groups[["image_id", "near_group_id", "group_size"]],
        on="image_id",
        how="left",
        suffixes=("", "_near"),
    )
    pool = merged[(merged["readable"]) & (merged["group_size"] == 1)].copy()
    if pool.empty:
        pool = merged[merged["readable"]].copy()

    picks = []
    for label in [0, 1]:
        class_pool = pool[pool["label"] == label].sort_values(["bytes", "rel_path"], ascending=[False, True])
        picks.extend(class_pool.head(3).to_dict("records"))

    tile = 230
    gap = 26
    top = 78
    left = 32
    label_h = 46
    width = left * 2 + 3 * tile + 2 * gap
    height = top + 2 * (tile + label_h) + gap + 24
    canvas = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (left, 22), "Representative ultrasound samples", TEXT)
    draw_label(draw, (left, 44), "Healthy/non-PCOS and PCOS-positive examples from readable dataset files", MUTED)

    for i, row in enumerate(picks):
        r = 0 if i < 3 else 1
        c = i % 3
        x = left + c * (tile + gap)
        y = top + r * (tile + label_h + gap)
        image = load_square(project_root / row["rel_path"], tile)
        title = "Healthy / non-PCOS" if row["label"] == 0 else "PCOS-positive"
        accent = HEALTHY if row["label"] == 0 else PCOS
        add_tile(canvas, image, (x, y), title, row["filename"], accent)

    path = output_dir / "dataset_sample_panel.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)
    return path


def build_duplicate_panel(project_root: Path, output_dir: Path) -> Path:
    examples = pd.read_csv(project_root / "reports/cross_label_phash_examples.csv")
    valid_groups = []
    for group_id, group in examples.groupby("near_group_id"):
        if set(group["label"]) == {0, 1}:
            healthy = group[group["label"] == 0].head(2)
            pcos = group[group["label"] == 1].head(2)
            if len(healthy) >= 1 and len(pcos) >= 1:
                valid_groups.append((group_id, pd.concat([healthy.head(1), pcos.head(1)])))
        if len(valid_groups) == 4:
            break

    tile = 210
    pair_gap = 20
    group_gap = 36
    top = 86
    left = 32
    label_h = 52
    width = left * 2 + 2 * (2 * tile + pair_gap) + group_gap
    height = top + 2 * (tile + label_h) + 38
    canvas = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (left, 22), "Cross-label pHash near-duplicate examples", TEXT)
    draw_label(draw, (left, 44), "Visually similar images appearing under conflicting labels; these groups were excluded from strict splits", MUTED)

    for idx, (group_id, group) in enumerate(valid_groups):
        r = idx // 2
        c = idx % 2
        x0 = left + c * (2 * tile + pair_gap + group_gap)
        y0 = top + r * (tile + label_h + 28)
        draw_label(draw, (x0, y0 - 22), group_id, TEXT)
        for j, (_, row) in enumerate(group.iterrows()):
            image = load_square(project_root / row["rel_path"], tile)
            title = "Healthy label" if row["label"] == 0 else "PCOS label"
            accent = HEALTHY if row["label"] == 0 else PCOS
            add_tile(canvas, image, (x0 + j * (tile + pair_gap), y0), title, Path(row["rel_path"]).name, accent)

    path = output_dir / "duplicate_conflict_panel.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)
    return path


def stack_existing_panels(paths: list[Path], title: str, subtitle: str, output_path: Path, max_width: int = 1500) -> Path:
    panels = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if image.width > max_width:
            h = int(image.height * max_width / image.width)
            image = image.resize((max_width, h), Image.Resampling.LANCZOS)
        panels.append(image)

    gap = 24
    header = 72
    width = max(panel.width for panel in panels) + 64
    height = header + sum(panel.height for panel in panels) + gap * (len(panels) - 1) + 32
    canvas = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (32, 20), title, TEXT)
    draw_label(draw, (32, 42), subtitle, MUTED)
    y = header
    for panel in panels:
        x = (width - panel.width) // 2
        draw.rectangle((x - 2, y - 2, x + panel.width + 1, y + panel.height + 1), outline=BORDER, width=2)
        canvas.paste(panel, (x, y))
        y += panel.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def build_gradcam_panels(project_root: Path, output_dir: Path) -> tuple[Path, Path]:
    comparison_paths = [
        project_root / "reports/gradcam_xai_comparison/xai_000_label0.png",
        project_root / "reports/gradcam_xai_comparison/xai_004_label1.png",
        project_root / "reports/gradcam_xai_comparison/xai_007_label1.png",
    ]
    sanity_paths = [
        project_root / "reports/gradcam_xai_sanity/supervised_resnet50pct_e5_sanity_003.png",
        project_root / "reports/gradcam_xai_sanity/simclr_resnet50pct_e10_sanity_003.png",
        project_root / "reports/gradcam_xai_sanity/supervised_resnet50pct_e5_sanity_004.png",
        project_root / "reports/gradcam_xai_sanity/simclr_resnet50pct_e10_sanity_004.png",
    ]
    comparison = stack_existing_panels(
        comparison_paths,
        "Grad-CAM comparison examples",
        "Representative healthy and PCOS-positive cases for supervised ResNet-18 and SimCLR checkpoints",
        output_dir / "gradcam_comparison_panel.png",
    )
    sanity = stack_existing_panels(
        sanity_paths,
        "Grad-CAM sanity and perturbation examples",
        "Trained, randomized, and border-masked explanations used to audit attribution reliability",
        output_dir / "gradcam_sanity_panel.png",
    )
    return comparison, sanity


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    paths = [
        build_dataset_panel(project_root, output_dir),
        build_duplicate_panel(project_root, output_dir),
    ]
    paths.extend(build_gradcam_panels(project_root, output_dir))
    Console().print("Wrote manuscript image panels:")
    for path in paths:
        Console().print(f"- {path}")


if __name__ == "__main__":
    main()
