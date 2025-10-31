from typing import Tuple, Union
from pathlib import Path
import hydra
import meshio
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pymathprim.geometry import laplacian, lumped_mass
from scipy.sparse import csr_matrix

from neural_cg.datagen_helper import DatagenBase


def to_tet_field(tets, field):
    out = np.zeros(tets.shape[0], dtype=field.dtype)
    for i in range(tets.shape[0]):
        out[i] = np.mean(field[tets[i]])
    return out.astype(field.dtype)


def get_laplacian(nodes, elem, random_field, var, rng, srf):
    if random_field:
        mesh = meshio.Mesh(nodes, {"triangle": elem})
        field = srf.mesh(mesh, points="points", seed=rng())
        field = field.astype(np.float64).copy()
        field = field - field.min()
        field = field / (field.max() + 1e-4)
        field = (field * var) + (1 - var)
        L = laplacian(nodes, elem, field)
    else:
        L = laplacian(nodes, elem)
    return L


class MultimeshHeatDatagen(DatagenBase):
    def __init__(self, config: DictConfig):
        self.path = Path(config.mesh_folder)
        self.eps = config.eps
        self.all_mesh = list(self.path.glob("*"))
        logger.info(f"Found {len(self.all_mesh)} meshes in {self.path}")
        config.basic.max_count = len(self.all_mesh)
        super().__init__(**config.basic)
        # check they are all files
        for mesh in self.all_mesh:
            if not mesh.is_dir():
                raise ValueError(f"Not a directory: {mesh}")

    def step(self) -> Tuple[csr_matrix, None, np.ndarray, Union[None, np.ndarray]]:
        idx = self.current_count
        mesh_dir = self.all_mesh[idx]
        nodes = np.load(mesh_dir / "vert_manifold.npy").astype(np.float64)
        elem = np.load(mesh_dir / "faces_manifold.npy").astype(np.int32)
        assert nodes.shape[1] == 3 and elem.shape[1] == 3, "Not a valid mesh file."
        L = laplacian(nodes, elem)
        M = lumped_mass(nodes, elem)
        S = csr_matrix(L + self.eps * M).sorted_indices()
        return S, None, nodes, None


@hydra.main(config_path="config", config_name="heat_objmesh", version_base="1.3")
def main(cfg):
    print(cfg)
    datagen = MultimeshHeatDatagen(cfg)
    datagen.generate()


if __name__ == "__main__":
    main()
