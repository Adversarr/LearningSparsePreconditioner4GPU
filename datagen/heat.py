from typing import Tuple, Union

import gstools as gs
import hydra
import meshio
import numpy as np
from gstools.random import MasterRNG
from loguru import logger
from pymathprim.geometry import laplacian, lumped_mass
from scipy.sparse import csr_matrix

from neural_cg.datagen_helper import DatagenBase, load_obj, tetrahedralize


def to_tet_field(tets, field):
    out = np.zeros(tets.shape[0], dtype=field.dtype)
    for i in range(tets.shape[0]):
        out[i] = np.mean(field[tets[i]])
    return out.astype(field.dtype)


class HeatDatagen(DatagenBase):
    def __init__(self, config):
        super().__init__(**config.basic)
        mesh_file = config.mesh_file
        self.nodes, self.elem = load_obj(mesh_file)
        self.nodes = self.nodes.astype(np.float64)
        self.elem = self.elem.astype(np.int32)
        if config.tetrahedralize.enable:
            self.nodes, self.elem = tetrahedralize(
                self.nodes,
                self.elem,
                config.tetrahedralize.visualize,
                config.tetrahedralize.switches,
            )
        n_nodes = self.nodes.shape[0]
        n_elems = self.elem.shape[0]
        physics = self.nodes.shape[1]
        intrisic = self.elem.shape[1] - 1
        logger.info(
            f"Heta Dataset for Single Mesh: {mesh_file} => n_nodes: {n_nodes}, n_elems: {n_elems}, "
            f"intrisic: {intrisic}, physics: {physics}"
        )

        self.rng = MasterRNG(config.rng)
        self.random_model = gs.Gaussian(dim=3, var=5, len_scale=1)
        self.srf = gs.SRF(self.random_model)
        if intrisic == 2:
            self.mesh = meshio.Mesh(self.nodes, {"triangle": self.elem})
        else:
            self.mesh = meshio.Mesh(self.nodes, {"tetra": self.elem})
        self.mass = lumped_mass(self.nodes, self.elem)
        logger.info(f"Mass.shape: {self.mass.shape}, nnz: {self.mass.nnz}")
        self.var = config.var
        self.eps = config.eps

        self.visualize = config.get("visualize", False)
        if self.visualize:
            import pyvista as pv
            self.plotter = pv.Plotter(notebook=False)
            elem_type = pv.CellType.TETRA if intrisic == 3 else pv.CellType.TRIANGLE
            self.grid = pv.UnstructuredGrid(
                {elem_type: self.elem},
                self.nodes,
            )
            self.grid.point_data["field"] = np.zeros(self.nodes.shape[0])
            self.plotter.add_mesh(
                self.grid,
                show_edges=True,
                scalars="field",
                color="white",
                opacity=1,
            )

    def get_shared(self):
        return self.nodes

    def topology(self) -> csr_matrix:
        return laplacian(self.nodes, self.elem).sorted_indices()

    def step(self) -> Tuple[csr_matrix, None, np.ndarray, Union[None, np.ndarray]]:
        field = self.srf.mesh(self.mesh, points="points", seed=self.rng())
        field = field.astype(np.float64).copy()
        field = field - field.min()
        field = field / (field.max() + 1e-4)
        field = (field * self.var) + (1 - self.var)
        assert np.all(field > 0)
        L = laplacian(self.nodes, self.elem, to_tet_field(self.elem, field))
        S = csr_matrix(L + self.eps * self.mass).sorted_indices()

        if self.visualize:
            self.grid.point_data["field"] = field
            self.plotter.update_scalars(field, render=False)
            self.plotter.show(interactive=False, auto_close=False)

        return S, None, field, None


@hydra.main(config_path="config", config_name="heat", version_base="1.3")
def main(cfg):
    print(cfg)
    datagen = HeatDatagen(cfg)
    datagen.generate()


if __name__ == "__main__":
    main()
