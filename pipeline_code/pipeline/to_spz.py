"""Stage 3 of the pipeline: compress a trained ``.ply`` splat into Niantic's
``.spz`` format (~10x smaller, with virtually no visible quality loss).

``.ply`` stays the canonical output; ``.spz`` is the web-friendly delivery file
for the Babylon.js / WebXR viewer.

This uses the official ``spz`` Python bindings (https://github.com/nianticlabs/spz).
They are not on PyPI, so install them once into the environment:

    uv pip install "spz @ git+https://github.com/nianticlabs/spz.git"

(Requires CMake + a C++ compiler.) If the bindings are not available, this
script prints instructions for the zero-install browser converter at
https://nianticlabs.github.io/spz and exits with code 2 so the rest of the
pipeline can continue -- the ``.ply`` is already a valid deliverable.

Coordinate systems: 3DGS ``.ply`` files are right-handed Y-down Z-forward (RDF);
web viewers (three.js / Babylon.js) expect RUB. By default we convert to RUB,
which matches what the ``.spz`` format stores and what the viewer expects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Distinct exit code so callers can tell "tool missing" from "real failure".
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SPZ_UNAVAILABLE = 2

_BROWSER_HINT = (
    "The spz Python bindings are not installed. Two options:\n"
    "  1. Zero-install: open https://nianticlabs.github.io/spz in a browser and\n"
    "     drag your .ply in to convert it to .spz.\n"
    "  2. Install the bindings (needs CMake + a C++ compiler), then re-run:\n"
    "       uv pip install \"spz @ git+https://github.com/nianticlabs/spz.git\""
)


def _coord(spz_module, name: str):
    """Map a coordinate-system name (e.g. 'RUB') to a spz enum value."""
    try:
        return getattr(spz_module.CoordinateSystem, name.upper())
    except AttributeError as exc:
        valid = [c for c in dir(spz_module.CoordinateSystem) if c.isupper()]
        raise argparse.ArgumentTypeError(
            f"Unknown coordinate system '{name}'. Valid: {', '.join(valid)}"
        ) from exc


def convert(input_ply: Path, output_spz: Path, from_coord: str, to_coord: str) -> int:
    if not input_ply.is_file():
        print(f"error: input .ply not found: {input_ply}", file=sys.stderr)
        return EXIT_ERROR
    if input_ply.suffix.lower() != ".ply":
        print(f"error: expected a .ply file, got: {input_ply}", file=sys.stderr)
        return EXIT_ERROR

    try:
        import spz
    except ImportError:
        print(_BROWSER_HINT, file=sys.stderr)
        return EXIT_SPZ_UNAVAILABLE

    output_spz.parent.mkdir(parents=True, exist_ok=True)

    # Load the PLY, converting into the target (web) coordinate system.
    unpack = spz.UnpackOptions()
    unpack.to_coord = _coord(spz, to_coord)
    cloud = spz.load_splat_from_ply(str(input_ply), unpack)

    # Save as compressed SPZ from that same coordinate system.
    pack = spz.PackOptions()
    pack.from_coord = _coord(spz, from_coord)
    spz.save_spz(cloud, pack, str(output_spz))

    ply_bytes = input_ply.stat().st_size
    spz_bytes = output_spz.stat().st_size
    ratio = ply_bytes / spz_bytes if spz_bytes else float("nan")
    print(
        f"Wrote {output_spz} "
        f"({cloud.num_points} gaussians, SH degree {cloud.sh_degree})"
    )
    print(
        f"Size: {ply_bytes / 1e6:.1f} MB -> {spz_bytes / 1e6:.1f} MB "
        f"({ratio:.1f}x smaller)"
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a trained 3DGS .ply into a compressed .spz.",
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Input .ply file produced by training.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .spz path (default: input path with .spz extension).",
    )
    parser.add_argument(
        "--to-coord", default="RUB", metavar="SYS",
        help="Coordinate system to convert the splat into (default: RUB, the "
             "three.js / Babylon.js convention).",
    )
    parser.add_argument(
        "--from-coord", default="RUB", metavar="SYS",
        help="Coordinate system the data is in when packing (default: RUB). "
             "Leave as default unless your viewer needs something else.",
    )
    args = parser.parse_args(argv)

    output = args.output or args.input.with_suffix(".spz")
    return convert(args.input, output, args.from_coord, args.to_coord)


if __name__ == "__main__":
    raise SystemExit(main())
