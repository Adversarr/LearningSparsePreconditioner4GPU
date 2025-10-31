from pathlib import Path
from typing import Tuple, Union

import gstools as gs
import hydra
import meshio
import numpy as np
from gstools.random import MasterRNG
from loguru import logger
from omegaconf import DictConfig
from pymathprim.geometry import laplacian, lumped_mass
from scipy.sparse import csr_matrix, diags

from neural_cg.datagen_helper import DatagenBase


def get_laplacian(
    nodes,
    elem,
    random_field,
    min_density: float,
    max_density: float,
    rng: MasterRNG,
    srf: gs.SRF,
):
    L = laplacian(nodes, elem)
    M = lumped_mass(nodes, elem).diagonal()
    if random_field:
        mesh = meshio.Mesh(nodes, {"tetra": elem})
        field = srf.mesh(mesh, points='points', seed=rng())
        field = field.astype(np.float64).copy()
        field = field - field.min()
        field = field / (field.max() + 1e-4)

        # make field piecewise constant
        field = np.where(field > 0.5, 1, 0)
        field = field * (max_density - min_density) + min_density
        field = field.astype(np.float64).copy()
    else:
        field = np.ones_like(M, dtype=np.float64) * min_density
    M = M * field

    if field.shape[0] < 1000 and False:  # For debugging, use PyVista to visualize small meshes
        import pyvista as pv
        mesh = meshio.Mesh(nodes, {"tetra": elem})
        mesh = pv.wrap(mesh)
        mesh.point_data["density"] = field
        
        # Set theme before plotting
        pv.set_plot_theme("document")
        
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, scalar_bar_args={'title': "Density"})
        plotter.show(interactive=False, window_size=(3840, 2160), screenshot=True, auto_close=True)
        plotter.screenshot("heat_tetmesh.png")  # Saves to PNG file
        exit()


    return L + diags(M), field


class MultimeshHeatDatagen(DatagenBase):
    def __init__(self, config: DictConfig):
        self.path = Path(config.mesh_folder)
        self.all_mesh = list(self.path.glob("*"))
        logger.info(f"Found {len(self.all_mesh)} meshes in {self.path}")
        use_all_mesh = config.get('use_all_mesh', True)
        if use_all_mesh:
            config.basic.max_count = len(self.all_mesh)
        if config.basic.max_count > len(self.all_mesh):
            raise ValueError(
                f"max_count {config.basic.max_count} is larger than the number of meshes {len(self.all_mesh)}"
            )
        super().__init__(**config.basic)
        # check they are all files
        for mesh in self.all_mesh:
            if not mesh.is_dir():
                raise ValueError(f"Not a directory: {mesh}")
        self.random_field = config.random_field
        self.min_density = config.min_density
        self.max_density = config.max_density
        self.var = config.get('var', 5)
        self.len_scale = config.get('len_scale', 1)
        self.rng = MasterRNG(config.rng)
        self.srf = gs.SRF(gs.Gaussian(dim=3, var=self.var, len_scale=self.len_scale))

    def step(self):
        idx = self.current_count
        mesh_dir = self.all_mesh[idx]
        nodes = np.load(mesh_dir / "vert_tetra.npy").astype(np.float64)
        elem = np.load(mesh_dir / "elems_tetra.npy").astype(np.int32)
        assert nodes.shape[1] == 3 and elem.shape[1] == 4, "Not a valid mesh file."
        S, field = get_laplacian(
            nodes,
            elem,
            self.random_field,
            self.min_density,
            self.max_density,
            self.rng,
            self.srf,
        )
        return S, None, nodes, None

@hydra.main(config_path="config", config_name="heat_tetmesh", version_base="1.3")
def main(cfg):
    print(cfg)
    datagen = MultimeshHeatDatagen(cfg)
    datagen.generate()


if __name__ == "__main__":
    main()