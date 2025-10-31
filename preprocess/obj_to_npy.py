import argparse
from pathlib import Path
from typing import List

from loguru import logger
import numpy as np
from tqdm import tqdm

from neural_cg.datagen_helper import load_obj, tetrahedralize


def save_vertices_faces(
    output_dir: Path, vertices: np.ndarray, faces: np.ndarray, suffix: str = ""
) -> None:
    """Save vertices and faces to numpy files."""
    np.save(output_dir / f"vert{suffix}.npy", vertices.astype(np.float64))
    np.save(output_dir / f"faces{suffix}.npy", faces.astype(np.int32))


def process_single_obj(obj_path: Path, output_dir: Path, do_tetra: bool) -> None:
    """Process a single OBJ file: save vertices/faces and optionally tetrahedralize."""
    obj_name = obj_path.stem

    # Stage 1: Load and save vertices/faces
    try:
        vertices, faces = load_obj(obj_path)
        specific_output = output_dir / obj_name
        specific_output.mkdir(parents=True, exist_ok=True)
        save_vertices_faces(specific_output, vertices, faces, "_manifold")
    except Exception as e:
        logger.error(f"Failed to process {obj_path}: {e}")
        return

    # Stage 2: Tetrahedralize if requested
    if do_tetra:
        try:
            vertices, faces = tetrahedralize(vertices, faces, False)
            save_vertices_faces(specific_output, vertices, faces, "_tetra")
        except Exception as e:
            logger.error(f"Tetrahedralization failed for {obj_path}: {e}")
            return


def find_objs(input_dir: Path) -> List[Path]:
    """Find all .obj files in a directory recursively."""
    return list(input_dir.glob("**/*.obj"))


def main():
    parser = argparse.ArgumentParser(
        description="""
            Process OBJ files to NPY format with optional tetrahedralization.

            Example usage:
                python obj_to_npy.py --input <input_dir> --output <output_dir> [--tetra]
        """
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing OBJ files."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for NPY files."
    )
    parser.add_argument(
        "--tetra", action="store_true", help="Enable tetrahedralization using tetgen."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List all OBJ files without processing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    assert input_dir.is_dir(), f"Input directory {input_dir} does not exist."
    output_dir = Path(args.output)

    objs = find_objs(input_dir)
    if args.list_only:
        print("Found OBJ files:")
        for obj in objs:
            print(obj)
        return

    pbar =  tqdm(objs, desc="Processing OBJ files")
    for obj_path in pbar:
        process_single_obj(obj_path, output_dir, args.tetra)
        pbar.set_postfix(
            {"current": obj_path.name, "total": len(objs)}, refresh=True
        )


if __name__ == "__main__":
    main()
