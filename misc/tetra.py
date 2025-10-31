from argparse import ArgumentParser
import numpy as np
from neural_cg.datagen_helper import tetrahedralize, load_obj

def main(cfg):
    obj_file = cfg.file
    verts, faces = load_obj(obj_file)
    tet_verts, tet_elems = tetrahedralize(verts, faces, False, switches=cfg.switches)
    print(tet_verts.shape)
    print(tet_elems.shape)
    np.save(cfg.output + "_verts.npy", tet_verts)
    np.save(cfg.output + "_elems.npy", tet_elems)
    print(f"Saved to {cfg.output}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the obj file")
    parser.add_argument("output", type=str, help="Path to the output file")
    parser.add_argument("--switches", type=str, default="pq1.1a6e-5", help="Switches for tetgen")
    args = parser.parse_args()
    main(args)