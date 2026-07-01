"""Stage 0 of the pipeline: compute camera poses with COLMAP for captures that
ship *without* a reconstruction.

Some SplatKing exports (e.g. the ``splatpack.v2`` / ``photo_dual`` format) contain
only the captured photos plus EXIF/IMU metadata -- there is **no** COLMAP model
(no ``cameras``/``images``/``points3D``, no camera positions, no point cloud). The
rest of the pipeline (``prepare_data.py`` -> training) needs that sparse model, so
this stage runs Structure-from-Motion to create it:

    feature_extractor  ->  exhaustive/sequential matcher  ->  mapper

The result is written in the same layout a SplatKing COLMAP export would have, so
``prepare_data.py`` consumes it unchanged:

    <output>/
    |-- images/                 the captured photos (flat)
    |-- sparse/0/
        |-- cameras.bin
        |-- images.bin
        |-- points3D.bin

Needs the COLMAP CLI on PATH (or under ``tools/colmap/``, or via ``--colmap-exe`` /
``$COLMAP_EXE``). A CUDA build uses the GPU for SIFT extraction and matching.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_data import _MODEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Files COLMAP writes / our own model layout -- never treat these as captures.
_COLMAP_BASENAMES = {"cameras", "images", "points3D"}


def find_colmap(explicit: str | Path | None = None) -> str:
    """Locate the COLMAP executable.

    Order: explicit arg, ``$COLMAP_EXE``, ``colmap`` on PATH, then a recursive
    search under ``tools/colmap/`` (where the install step unpacks the Windows
    build).
    """
    for candidate in (explicit, os.environ.get("COLMAP_EXE")):
        if candidate:
            p = Path(candidate)
            if p.is_file():
                return str(p)

    on_path = shutil.which("colmap")
    if on_path:
        return on_path

    tools = PROJECT_ROOT / "tools" / "colmap"
    if tools.is_dir():
        exe = "colmap.exe" if os.name == "nt" else "colmap"
        for hit in tools.rglob(exe):
            if hit.is_file():
                return str(hit)

    raise FileNotFoundError(
        "COLMAP executable not found. Install it (see README, Stage 0) or pass "
        "--colmap-exe / set $COLMAP_EXE. Looked on PATH and under tools/colmap/."
    )


def _run(colmap: str, command: str, args: list[str], dry_run: bool = False) -> None:
    cmd = [colmap, command, *args]
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_quality(input_dir: Path) -> dict[str, tuple[str, float]]:
    """Map image filename -> (quality_band, quality_score) from quality_flags.csv.

    Returns an empty dict if the file is absent (then no filtering happens).
    """
    out: dict[str, tuple[str, float]] = {}
    for csv_path in input_dir.rglob("quality_flags.csv"):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("filename") or "").strip()
                if not name:
                    continue
                band = (row.get("quality_band") or "").strip().lower()
                try:
                    score = float(row.get("quality_score") or "nan")
                except ValueError:
                    score = float("nan")
                out[name] = (band, score)
        break  # one capture folder == one quality file
    return out


def _collect_images(
    input_dir: Path,
    dst: Path,
    min_quality: float,
    drop_bands: set[str],
) -> int:
    """Copy capture photos into ``dst`` (flat), applying optional quality filters."""
    quality = _load_quality(input_dir) if (min_quality > 0 or drop_bands) else {}

    sources: list[Path] = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and p.stem not in _COLMAP_BASENAMES
    )
    if not sources:
        raise FileNotFoundError(f"No .jpg/.png photos found under {input_dir}.")

    dst.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    for src in sources:
        band, score = quality.get(src.name, ("", float("nan")))
        if drop_bands and band in drop_bands:
            skipped += 1
            continue
        if min_quality > 0 and score == score and score < min_quality:  # score==score: not NaN
            skipped += 1
            continue
        shutil.copy2(src, dst / src.name)
        kept += 1

    if skipped:
        print(f"Quality filter : kept {kept}, skipped {skipped} (low quality).")
    return kept


def _model_stats(colmap: str, model_dir: Path) -> dict[str, int]:
    """Run model_analyzer and parse a few integer stats (best-effort)."""
    try:
        proc = subprocess.run(
            [colmap, "model_analyzer", "--path", str(model_dir)],
            capture_output=True, text=True, check=False,
        )
    except Exception:  # pragma: no cover - analyzer is only informational
        return {}
    text = f"{proc.stdout}\n{proc.stderr}"
    stats: dict[str, int] = {}
    for key, pat in (
        ("cameras", r"Cameras:\s*(\d+)"),
        ("images", r"(?:Registered images|Images):\s*(\d+)"),
        ("points", r"Points:\s*(\d+)"),
    ):
        m = re.search(pat, text)
        if m:
            stats[key] = int(m.group(1))
    return stats


def _convert_to_txt(colmap: str, model_dir: Path, dry_run: bool = False) -> None:
    """Convert a COLMAP model to TXT in place and drop the .bin files.

    The pycolmap reader that gsplat uses (rmbrualla fork) mis-reads .bin headers
    on Windows -- it does ``struct.unpack('L', f.read(8))`` assuming an 8-byte
    long, but ``long`` is 4 bytes in MSVC. The TXT model avoids that entirely.
    """
    _run(colmap, "model_converter", [
        "--input_path", str(model_dir),
        "--output_path", str(model_dir),
        "--output_type", "TXT",
    ], dry_run=dry_run)
    if dry_run:
        return
    for name in _MODEL_NAMES:
        bin_file = model_dir / f"{name}.bin"
        if bin_file.exists() and (model_dir / f"{name}.txt").exists():
            bin_file.unlink()


def _select_best_model(colmap: str, sparse_dir: Path) -> Path:
    """COLMAP may produce several sub-models (0,1,..). Keep the one with the most
    registered images as ``sparse/0`` and drop the rest, so downstream is
    unambiguous. Returns the final ``sparse/0`` path.
    """
    submodels = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
    if not submodels:
        raise FileNotFoundError(
            f"COLMAP mapper produced no model under {sparse_dir}. The capture "
            "likely has too little overlap/texture to reconstruct."
        )

    best = max(submodels, key=lambda d: _model_stats(colmap, d).get("images", 0))
    final = sparse_dir / "0"

    if best != final:
        # Move the loser '0' out of the way, then promote the best one to '0'.
        if final.exists():
            tmp = sparse_dir / "_old0"
            final.rename(tmp)
            if best == final:  # pragma: no cover - guarded above
                best = tmp
        best.rename(final)

    for d in list(sparse_dir.iterdir()):
        if d.is_dir() and d != final:
            shutil.rmtree(d)
    return final


def run_colmap(
    input_dir: Path,
    work_dir: Path,
    *,
    colmap_exe: str | Path | None = None,
    matcher: str = "exhaustive",
    camera_model: str = "OPENCV",
    use_gpu: bool = True,
    min_quality: float = 0.0,
    drop_bands: set[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    """Run COLMAP SfM on a raw capture folder.

    Returns ``work_dir`` containing ``images/`` and ``sparse/0/`` -- the layout
    ``prepare_data.prepare`` expects as its input.
    """
    input_dir = Path(input_dir)
    work_dir = Path(work_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {input_dir}")

    colmap = find_colmap(colmap_exe)
    print(f"COLMAP       : {colmap}")

    images_dir = work_dir / "images"
    sparse_dir = work_dir / "sparse"
    database = work_dir / "database.db"

    if work_dir.exists() and any(work_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Work dir {work_dir} is not empty. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    n = _collect_images(input_dir, images_dir, min_quality, drop_bands or set())
    print(f"Images       : {images_dir}  ({n} photos)")

    gpu = "1" if use_gpu else "0"

    _run(colmap, "feature_extractor", [
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        # One camera per image, each seeded from EXIF focal length. This is the
        # robust choice for a dual-lens capture (ultra-wide + wide have different
        # intrinsics) -- COLMAP reads the EXIF automatically.
        "--ImageReader.single_camera", "0",
        # COLMAP 4.x renamed the SIFT option prefixes to FeatureExtraction/FeatureMatching.
        "--FeatureExtraction.use_gpu", gpu,
    ], dry_run=dry_run)

    match_cmd = {
        "exhaustive": "exhaustive_matcher",
        "sequential": "sequential_matcher",
    }.get(matcher)
    if match_cmd is None:
        raise ValueError(f"Unknown matcher '{matcher}' (use exhaustive|sequential).")
    _run(colmap, match_cmd, [
        "--database_path", str(database),
        "--FeatureMatching.use_gpu", gpu,
    ], dry_run=dry_run)

    sparse_dir.mkdir(parents=True, exist_ok=True)
    _run(colmap, "mapper", [
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ], dry_run=dry_run)

    if dry_run:
        print("\n(dry run: skipping model selection / stats)")
        return work_dir

    final = _select_best_model(colmap, sparse_dir)
    # TXT instead of .bin so the (Windows-buggy) pycolmap reader can load it.
    _convert_to_txt(colmap, final)
    stats = _model_stats(colmap, final)
    print("\n=== COLMAP reconstruction ===")
    print(f"Model        : {final}")
    print(
        f"Registered   : {stats.get('images', '?')} / {n} images, "
        f"{stats.get('cameras', '?')} cameras, {stats.get('points', '?')} 3D points"
    )
    reg = stats.get("images", 0)
    if n and reg and reg < 0.6 * n:
        print(
            f"warning: only {reg}/{n} images registered. Consider --matcher sequential, "
            "dropping blurry frames with --min-quality, or re-capturing with more overlap.",
            file=sys.stderr,
        )
    print(f"\nSfM ready: {work_dir}  (feed this to prepare_data.py)")
    return work_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM on a raw capture (no poses) -> images/ + sparse/0/.",
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Folder with the captured photos (e.g. a SplatKing photo_dual export).",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path,
        help="Work folder to write images/ and sparse/0/ into.",
    )
    parser.add_argument(
        "--matcher", choices=("exhaustive", "sequential"), default="exhaustive",
        help="Feature matcher. 'exhaustive' (default) is best for small sets; "
             "'sequential' suits ordered video-like captures.",
    )
    parser.add_argument(
        "--camera-model", default="OPENCV",
        help="COLMAP camera model (default OPENCV, models lens distortion).",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.0,
        help="Drop frames whose quality_score (from quality_flags.csv) is below "
             "this value. Default 0.0 = keep all.",
    )
    parser.add_argument(
        "--drop-bands", default="",
        help="Comma-separated quality_band values to drop (e.g. 'poor'). "
             "Default: none.",
    )
    parser.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false",
        help="Disable GPU SIFT (use for a CPU-only COLMAP build).",
    )
    parser.add_argument(
        "--colmap-exe", type=Path, default=None,
        help="Path to the COLMAP executable (else $COLMAP_EXE / PATH / tools/colmap).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the output folder if it already contains files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the COLMAP commands without running them.",
    )
    args = parser.parse_args(argv)

    drop_bands = {b.strip().lower() for b in args.drop_bands.split(",") if b.strip()}
    try:
        run_colmap(
            input_dir=args.input,
            work_dir=args.output,
            colmap_exe=args.colmap_exe,
            matcher=args.matcher,
            camera_model=args.camera_model,
            use_gpu=args.use_gpu,
            min_quality=args.min_quality,
            drop_bands=drop_bands,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: COLMAP step failed (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
