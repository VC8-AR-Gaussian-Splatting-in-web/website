"""Stage 1 of the pipeline: turn a SplatKing capture export into the dataset
layout that the gsplat trainer expects.

SplatKing (LiDAR capture) exports a COLMAP text model -- ``cameras``, ``images``
and ``points3D`` files plus the captured photos. gsplat's COLMAP parser expects:

    <output>/
    |-- images/                 full-resolution photos
    |-- images_2/  (optional)   half-resolution   (use with --data_factor 2)
    |-- images_4/  (optional)   quarter-resolution (use with --data_factor 4)
    |-- sparse/0/
        |-- cameras.{bin,txt}
        |-- images.{bin,txt}
        |-- points3D.{bin,txt}

This script locates those pieces anywhere inside the export, copies them into
that layout, and optionally writes down-sampled image folders so training fits
in limited VRAM (the RTX 4070 Super has 12 GB).

Runs anywhere (macOS or Windows) -- only needs Pillow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

# COLMAP sparse-model base names (the "images" model file holds camera poses,
# which is distinct from the folder of photos).
_MODEL_NAMES = ("cameras", "images", "points3D")
_MODEL_EXTS = (".bin", ".txt")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _find_model_dir(root: Path) -> Path:
    """Return the directory holding a complete COLMAP sparse model."""
    candidates: dict[Path, set[str]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in _MODEL_EXTS and path.stem in _MODEL_NAMES:
            candidates.setdefault(path.parent, set()).add(path.stem)

    complete = [d for d, names in candidates.items() if set(_MODEL_NAMES) <= names]
    if not complete:
        raise FileNotFoundError(
            "No complete COLMAP model (cameras/images/points3D) found under "
            f"{root}. In SplatKing, export the capture as a COLMAP model."
        )
    # Prefer the shallowest match (e.g. an explicit sparse/0 over a stray copy).
    complete.sort(key=lambda d: len(d.parts))
    return complete[0]


def _find_images_dir(root: Path, model_dir: Path) -> Path:
    """Return the directory that holds the captured photos."""
    counts: dict[Path, int] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            counts[path.parent] = counts.get(path.parent, 0) + 1

    # Ignore any down-sampled folders that might already exist in the export.
    counts = {d: n for d, n in counts.items() if d != model_dir}
    if not counts:
        raise FileNotFoundError(f"No .jpg/.png photos found under {root}.")

    # The capture folder is the one with the most photos.
    return max(counts, key=counts.get)


def _copy_model(model_dir: Path, dst: Path) -> list[str]:
    """Copy the COLMAP model files into ``dst`` (preferring .bin per name)."""
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in _MODEL_NAMES:
        src = None
        for ext in _MODEL_EXTS:  # .bin preferred over .txt
            candidate = model_dir / f"{name}{ext}"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            raise FileNotFoundError(f"Missing COLMAP file '{name}' in {model_dir}.")
        shutil.copy2(src, dst / src.name)
        copied.append(src.name)
    return copied


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _place_full_res(images: list[Path], dst: Path, mode: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src in images:
        target = dst / src.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if mode == "copy":
            shutil.copy2(src, target)
        elif mode == "move":
            shutil.move(str(src), str(target))
        elif mode == "symlink":
            target.symlink_to(src.resolve())
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"Unknown link mode: {mode}")


def _write_downsampled(images: list[Path], out_root: Path, factor: int) -> None:
    """Write images_{factor}/ at 1/factor resolution (LANCZOS)."""
    dst = out_root / f"images_{factor}"
    dst.mkdir(parents=True, exist_ok=True)
    for src in images:
        with Image.open(src) as im:
            w, h = im.size
            new_size = (max(1, round(w / factor)), max(1, round(h / factor)))
            im.resize(new_size, Image.LANCZOS).save(dst / src.name)


def prepare(
    input_dir: Path,
    output_dir: Path,
    downsample: list[int],
    link_mode: str,
    overwrite: bool,
) -> None:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {input_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output {output_dir} is not empty. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    model_dir = _find_model_dir(input_dir)
    images_dir = _find_images_dir(input_dir, model_dir)
    images = _list_images(images_dir)

    print(f"COLMAP model : {model_dir}")
    print(f"Photos       : {images_dir}  ({len(images)} images)")

    copied = _copy_model(model_dir, output_dir / "sparse" / "0")
    print(f"Wrote model  : sparse/0/  ({', '.join(copied)})")

    _place_full_res(images, output_dir / "images", link_mode)
    print(f"Wrote images : images/  ({link_mode})")

    for factor in downsample:
        _write_downsampled(images, output_dir, factor)
        print(f"Wrote images : images_{factor}/  (1/{factor} resolution)")

    print(f"\nDataset ready: {output_dir}")
    if downsample:
        factors = ", ".join(str(f) for f in downsample)
        print(f"Train with --data_factor in {{1, {factors}}} (higher = less VRAM).")


def _parse_downsample(value: str) -> list[int]:
    if not value.strip():
        return []
    factors = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        f = int(chunk)
        if f < 2:
            raise argparse.ArgumentTypeError("downsample factors must be >= 2")
        factors.append(f)
    return sorted(set(factors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arrange a SplatKing COLMAP export into the gsplat dataset layout.",
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Folder containing the SplatKing export (images + COLMAP model).",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path,
        help="Destination dataset folder for training (e.g. data/my_scene).",
    )
    parser.add_argument(
        "--downsample", type=_parse_downsample, default=[2, 4],
        metavar="FACTORS",
        help="Comma-separated extra resolutions to generate, e.g. '2,4'. "
             "Empty string disables. Default: 2,4.",
    )
    parser.add_argument(
        "--link-mode", choices=("copy", "symlink", "move"), default="copy",
        help="How to place full-res images. 'copy' is safest for cross-machine "
             "transfer (default); 'symlink' saves disk on the same machine.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the output folder if it already contains files.",
    )
    args = parser.parse_args(argv)

    try:
        prepare(
            input_dir=args.input,
            output_dir=args.output,
            downsample=args.downsample,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, NotADirectoryError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
