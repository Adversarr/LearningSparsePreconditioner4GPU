from pathlib import Path 

import hydra
from loguru import logger
import numpy as np
from pymathprim.geometry import laplacian
import matplotlib.pyplot as plt
from neural_cg.datagen_helper import (
    DatagenBase,
    extract_boundary_faces,
    faces_to_edges,
    extract_boundary_vertices,
    subgraph_grow
)

class MultimeshPoissonDatagen(DatagenBase):
    def __init__(self, config):
        self.path = Path(config.mesh_folder)
        self.all_mesh = list(self.path.glob("*"))
        logger.info(f"Found {len(self.all_mesh)} meshes in {self.path}")
        config.basic.max_count = len(self.all_mesh)
        super().__init__(**config.basic)
        # mesh_file = config.mesh_file

        # check they are all files
        for mesh in self.all_mesh:
            if not mesh.is_dir():
                raise ValueError(f"Not a directory: {mesh}")

        self.seed_count = config.seed_count
        self.grow_iteration = config.grow_iteration

        self.visualize = config.visualize
        if config.visualize:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

    def step(self):
        idx = self.current_count
        mesh_dir = self.all_mesh[idx]
        nodes = np.load(mesh_dir / "vert_tetra.npy").astype(np.float64)
        elem = np.load(mesh_dir / "elems_tetra.npy").astype(np.int32)
        boundary_nodes = np.array(extract_boundary_vertices(elem))
        boundary_faces = np.array(extract_boundary_faces(elem))
        boundary_edges = faces_to_edges(boundary_faces)
        
        n_nodes = nodes.shape[0]
        n_elems = elem.shape[0]
        physics = nodes.shape[1]
        intrisic = elem.shape[1] - 1
        logger.info(
            f"Heta Dataset for Single Mesh: {mesh_dir} => n_nodes: {n_nodes}, n_elems: {n_elems}, "
            f"intrisic: {intrisic}, physics: {physics}"
        )
        assert len(boundary_nodes) > self.seed_count
        logger.info(f"Boundary shape: {boundary_nodes.shape}, seed count: {self.seed_count}")
        

        L = laplacian(nodes, elem)
        dbc = np.random.choice(boundary_nodes.size, size=self.seed_count, replace=False)
        dbc = boundary_nodes[dbc]
        dbc = subgraph_grow(n_nodes, dbc, boundary_edges, self.grow_iteration)
        mask = np.ones(nodes.shape[0], dtype=np.float64)
        mask[dbc] = 0

        if self.visualize:
            self.ax.cla()
            # dbc nodes as red
            nbc = np.setdiff1d(np.arange(n_nodes), np.arange(n_nodes)[dbc])
            self.ax.scatter(nodes[nbc, 0], nodes[nbc, 1], nodes[nbc, 2], c='b', s=1)
            # boundary nodes as blue
            self.ax.scatter(nodes[dbc, 0], nodes[dbc, 1], nodes[dbc, 2], c='r', s=1)
            plt.pause(0.01)
 
        return L, mask, nodes, None

@hydra.main(config_path="config", config_name="poisson3d_tetmesh", version_base="1.3")
def main(cfg):
    print(cfg)
    datagen = MultimeshPoissonDatagen(cfg)
    datagen.generate()

if __name__ == "__main__":
    main()