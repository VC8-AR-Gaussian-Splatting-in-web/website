"""End-to-end pipeline: captured images -> trained splat -> .ply (+ optional .spz).

This drives the three stages:

  1. prepare_data.py  -- arrange a SplatKing export into the gsplat layout
  2. gsplat training  -- examples/simple_trainer.py (needs Windows + CUDA GPU)
  3. to_spz.py        -- compress the resulting .ply into .spz (optional)

Run it from the project root inside the uv environment, e.g.:

    uv run python pipeline/run_pipeline.py --scene chair \\
        --input ~/captures/chair_splatking

The training stage shells out to the gsplat trainer, which must already be
installed in this environment (see README). prepare/convert run anywhere; only
the training stage requires the CUDA GPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# pipeline/ is on sys.path[0] when run as a script, so sibling imports work.
from prepare_data import prepare
from run_colmap import run_colmap
from to_spz import EXIT_SPZ_UNAVAILABLE, convert

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _reexec_in_venv() -> None:
    """Re-run under the project's .venv Python if started with another interpreter.

    The training stage shells out with ``sys.executable``; if the user launched this
    with the system Python (no venv active), that subprocess would miss tyro/torch/etc.
    Re-exec'ing with the venv interpreter makes ``python pipeline/run_pipeline.py`` work
    regardless of which Python is on PATH.
    """
    scripts = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    venv_py = PROJECT_ROOT / ".venv" / scripts / exe
    if not venv_py.exists():
        return
    try:
        if os.path.samefile(sys.executable, venv_py):
            return
    except OSError:
        pass
    print(f"(switching to project venv: {venv_py})", flush=True)
    os.execv(str(venv_py), [str(venv_py), *sys.argv])


def _run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(str(c) for c in cmd)
    location = f"  (cwd: {cwd})" if cwd else ""
    print(f"\n$ {printable}{location}\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _find_latest_ply(ply_dir: Path) -> Path | None:
    """Return the point_cloud_<step>.ply with the highest step, if any."""
    best: tuple[int, Path] | None = None
    for ply in ply_dir.glob("point_cloud_*.ply"):
        stem = ply.stem.rsplit("_", 1)[-1]
        if not stem.isdigit():
            continue
        step = int(stem)
        if best is None or step > best[0]:
            best = (step, ply)
    return best[1] if best else None


def _build_train_cmd(
    *,
    gsplat_dir: Path,
    data_dir: Path,
    result_dir: Path,
    data_factor: int,
    max_steps: int,
    sh_degree: int,
    viewer: bool,
    extra: list[str],
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "simple_trainer.py",
        "default",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--data_factor", str(data_factor),
        "--max_steps", str(max_steps),
        "--sh_degree", str(sh_degree),
        "--save_ply",
    ]
    if not viewer:
        cmd.append("--disable_viewer")
    cmd += extra
    return cmd


def run(args: argparse.Namespace) -> int:
    gsplat_dir = args.gsplat_dir.resolve()
    examples_dir = gsplat_dir / "examples"
    trainer = examples_dir / "simple_trainer.py"

    data_dir = (args.data_dir or PROJECT_ROOT / "data" / args.scene).resolve()
    result_dir = (args.result_dir or PROJECT_ROOT / "results" / args.scene).resolve()
    output_ply = (args.output or result_dir / f"{args.scene}.ply").resolve()

    # --- Stage 0: COLMAP SfM (optional, for captures without poses) ----------
    prepare_input = args.input
    if args.colmap:
        if args.input is None:
            print(
                "error: --colmap needs --input (the raw capture folder).",
                file=sys.stderr,
            )
            return 1
        print("=== Stage 0: COLMAP (computing camera poses) ===")
        sfm_dir = (data_dir.parent / f"{args.scene}_sfm").resolve()
        drop_bands = {b.strip().lower() for b in args.drop_bands.split(",") if b.strip()}
        run_colmap(
            input_dir=args.input.resolve(),
            work_dir=sfm_dir,
            colmap_exe=args.colmap_exe,
            matcher=args.matcher,
            min_quality=args.min_quality,
            drop_bands=drop_bands,
            use_gpu=not args.no_gpu,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        prepare_input = sfm_dir

    # --- Stage 1: prepare (only if a raw capture export was given) -----------
    if args.colmap and args.dry_run:
        print("(dry run: skipping prepare; COLMAP + dataset would be built here)")
    elif prepare_input is not None:
        print("=== Stage 1/3: preparing dataset ===")
        prepare(
            input_dir=Path(prepare_input).resolve(),
            output_dir=data_dir,
            downsample=args.downsample,
            link_mode="copy",
            overwrite=args.overwrite,
        )
    else:
        print("=== Stage 1/3: skipped (using existing --data-dir) ===")
        if not (data_dir / "sparse" / "0").is_dir():
            print(
                f"error: {data_dir} is not a prepared dataset (no sparse/0/). "
                "Pass --input to prepare one, or fix --data-dir.",
                file=sys.stderr,
            )
            return 1

    # --- Stage 2: train ------------------------------------------------------
    print("=== Stage 2/3: training (gsplat) ===")
    if not trainer.is_file() and not args.dry_run:
        print(
            f"error: gsplat trainer not found at {trainer}.\n"
            "Clone and install gsplat first (see README), or point --gsplat-dir "
            "at your checkout.",
            file=sys.stderr,
        )
        return 1

    train_cmd = _build_train_cmd(
        gsplat_dir=gsplat_dir,
        data_dir=data_dir,
        result_dir=result_dir,
        data_factor=args.data_factor,
        max_steps=args.max_steps,
        sh_degree=args.sh_degree,
        viewer=args.viewer,
        extra=args.extra,
    )
    try:
        _run(train_cmd, cwd=examples_dir, dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        print(f"error: training failed (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode

    if args.dry_run:
        print("\n(dry run: skipping output collection)")
        return 0

    # --- Stage 3: collect .ply and optionally compress to .spz ---------------
    print("=== Stage 3/3: collecting outputs ===")
    latest_ply = _find_latest_ply(result_dir / "ply")
    if latest_ply is None:
        print(
            f"error: no point_cloud_*.ply found in {result_dir / 'ply'}. "
            "Did training reach a ply_steps checkpoint?",
            file=sys.stderr,
        )
        return 1

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    output_ply.write_bytes(latest_ply.read_bytes())
    print(f"Final model : {output_ply}  (from {latest_ply.name})")

    if args.spz:
        output_spz = output_ply.with_suffix(".spz")
        code = convert(output_ply, output_spz, from_coord="RUB", to_coord="RUB")
        if code == EXIT_SPZ_UNAVAILABLE:
            print(
                "note: skipped .spz (bindings unavailable). The .ply above is a "
                "valid deliverable; see to_spz.py for the browser converter.",
            )
        elif code != 0:
            return code

    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    _reexec_in_venv()
    parser = argparse.ArgumentParser(
        description="Run the images -> splat -> .ply/.spz pipeline end to end.",
    )
    parser.add_argument(
        "--scene", required=True,
        help="Short scene name; drives default data/ and results/ paths.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="SplatKing export folder. If given, the dataset is prepared first.",
    )
    parser.add_argument(
        "--colmap", action="store_true",
        help="Run COLMAP SfM first (Stage 0) to compute poses for a raw capture "
             "with no COLMAP model (e.g. a SplatKing photo_dual export).",
    )
    parser.add_argument(
        "--matcher", choices=("exhaustive", "sequential"), default="exhaustive",
        help="COLMAP feature matcher for --colmap (default exhaustive).",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.0,
        help="--colmap: drop frames below this quality_score (default 0 = keep all).",
    )
    parser.add_argument(
        "--drop-bands", default="",
        help="--colmap: comma-separated quality_band values to drop, e.g. 'poor'.",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="--colmap: disable GPU SIFT (for a CPU-only COLMAP build).",
    )
    parser.add_argument(
        "--colmap-exe", type=Path, default=None,
        help="Path to the COLMAP executable (else $COLMAP_EXE / PATH / tools/colmap).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Prepared dataset folder (default: data/<scene>).",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=None,
        help="Training output folder (default: results/<scene>).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Final .ply path (default: results/<scene>/<scene>.ply).",
    )
    parser.add_argument(
        "--gsplat-dir", type=Path, default=PROJECT_ROOT / "third_party" / "gsplat",
        help="Path to the gsplat checkout (default: third_party/gsplat).",
    )
    parser.add_argument(
        "--data-factor", type=int, default=1,
        help="Train on 1/N resolution images (needs images_N/ from prepare). "
             "Default 1 (full res).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30_000,
        help="Training iterations (default 30000).",
    )
    parser.add_argument(
        "--sh-degree", type=int, default=3,
        help="Spherical-harmonics degree (default 3 = best quality).",
    )
    parser.add_argument(
        "--downsample", type=lambda s: [int(x) for x in s.split(",") if x.strip()],
        default=[2, 4],
        help="Extra resolutions to generate during prepare (default: 2,4).",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Open the live training viewer (default: headless).",
    )
    parser.add_argument(
        "--no-spz", dest="spz", action="store_false",
        help="Do not attempt .ply -> .spz conversion.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite the prepared dataset folder if it exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the training command without running it.",
    )
    parser.add_argument(
        "extra", nargs="*",
        help="Extra args passed through to simple_trainer.py (after '--').",
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
