import argparse
from pathlib import Path
from typing import List

from loguru import logger
import numpy as np
from tqdm import tqdm

from neural_cg.datagen_helper import load_msh


def save_vertices_elems(
    output_dir: Path,
    vertices: np.ndarray,
    elems: np.ndarray,
    suffix: str = "",
) -> None:
    """Save vertices and elems to numpy files."""
    np.save(output_dir / f"vert{suffix}.npy", vertices.astype(np.float64))
    np.save(output_dir / f"elems{suffix}.npy", elems.astype(np.int32))


def process_single_msh(
    msh_path: Path, output_dir: Path, min_size: int, max_size: int
) -> int:
    """Process a single MSH file: save vertices/elems and optionally tetrahedralize."""
    msh_name = msh_path.stem

    # Stage 1: Load and save vertices/elems
    try:
        vertices, elems = load_msh(msh_path)
        # remove all the mean and make it unit length approximately.
        vertices = vertices - np.mean(vertices, axis=0)
        vertices = vertices / np.max(np.abs(vertices))


        # Validate mesh size
        if len(vertices) < min_size or len(vertices) > max_size:
            # logger.warning(
            #     f"Skipping {msh_path}: vertex count {len(vertices)} is outside the allowed range [{min_size}, {max_size}]"
            # )
            return 0

        specific_output = output_dir / msh_name
        specific_output.mkdir(parents=True, exist_ok=True)
        save_vertices_elems(specific_output, vertices, elems, "_tetra")
        return 1
    except Exception as e:
        logger.error(f"Failed to process {msh_path}: {e}")
        return 0


def find_mshs(input_dir: Path) -> List[Path]:
    """Find all .msh files in a directory recursively."""
    return list(input_dir.glob("**/*.msh"))


def main():
    parser = argparse.ArgumentParser(
        description="""
            Process MSH files to NPY format with optional tetrahedralization.

            Example usage:
                python msh_to_npy.py --input <input_dir> --output <output_dir> [--tetra]
        """
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing MSH files."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for NPY files."
    )
    parser.add_argument(
        "--tetra", action="store_true", help="Enable tetrahedralization using tetgen."
    )
    parser.add_argument(
        "--max-mesh-size",
        type=int,
        default=32000,
        help="Maximum allowed vertex count in a mesh.",
    )
    parser.add_argument(
        "--min-mesh-size",
        type=int,
        default=400,
        help="Minimum allowed vertex count in a mesh.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List all MSH files without processing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    assert input_dir.is_dir(), f"Input directory {input_dir} does not exist."
    output_dir = Path(args.output)

    mshs = find_mshs(input_dir)
    if args.list_only:
        print("Found MSH files:")
        for msh in mshs:
            print(msh)
        print(f"Total: {len(mshs)}")
        return

    pbar = tqdm(mshs, desc="Processing MSH files")
    cnt = 0
    for msh_path in pbar:
        cnt += process_single_msh(
            msh_path, output_dir, args.min_mesh_size, args.max_mesh_size
        )
        pbar.set_postfix({"current": msh_path.name, "total": cnt})


if __name__ == "__main__":
    main()
