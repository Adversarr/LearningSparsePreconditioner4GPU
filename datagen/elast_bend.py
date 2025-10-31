import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyssim.fem import TetFiniteElementSolver_Host, unit_box

from neural_cg.datagen_helper import DatagenBase
from matplotlib import pyplot as plt
import pyvista as pv


def rotate_around_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R


class Elasticity(DatagenBase):
    def __init__(self, config):
        super().__init__(**config.basic)
        resolution: int = config.resolution
        self.nx = nx = config.nx
        vert, elem = unit_box(resolution * nx, resolution, resolution)
        vert = vert.T.copy()
        elem = elem.T.copy()
        vert[:, 0] *= nx
        vert[:, 1] -= 0.5
        vert[:, 2] -= 0.5

        self.vert = vert
        self.elem = elem
        self.time_step = config.time_step
        self.density = config.density
        self.youngs = config.youngs
        self.poisson = config.poisson
        logger.info(f"vert: {vert.shape}, elem: {elem.shape}, ")

        tsolver = TetFiniteElementSolver_Host(
            vert,
            elem,
            method="lbfgs_pd",
            time_step=self.time_step,
            young_modulus=self.youngs,
            poisson_ratio=self.poisson,
            density=self.density,
        )

        tsolver.set_rtol(1e-5)
        tsolver.add_gravity(np.array([0, 0, -9.8], dtype=np.float64))
        left_bc = vert[:, 0] == vert[:, 0].min()
        self.left_bc_dofs = left_bc_dofs = np.where(left_bc)[0]
        left_bc_deform = np.zeros((left_bc_dofs.size, 3), dtype=np.float64).T
        tsolver.mark_dirichlet_batched(left_bc_dofs, left_bc_deform)
        tsolver.reset()
        self.nV = vert.shape[0]
        self.nE = elem.shape[0]
        self.all_bc_idx = np.concatenate([left_bc_dofs, ])
        logger.info(f"Length of bc = {len(self.all_bc_idx)}")

        self.solver = tsolver
        self.T = 0
        self.visualize = config.visualize

        if self.visualize:
            self.plotter = pv.Plotter(notebook=False)
            self.mesh = pv.UnstructuredGrid(
                {pv.CellType.TETRA: self.elem},
                self.vert
            )
            # set camera
            self.plotter.camera.viewup = [0, 0, 1]
            self.plotter.camera.clipping_range = [1e-2, 10]
            self.plotter.camera.focal_point = [self.nx / 2, 0, 0]
            self.plotter.camera.position = [self.nx / 2, self.nx + 5, 1]
            self.plotter.add_mesh(self.mesh, show_edges=True)
            self.plotter.open_gif("bend.gif", fps=25)
            # self.plotter.show(interactive=False, auto_close=False)

    def topology(self):
        # Return the topology of the mesh
        self.solver.update_hessian_unfiltered(False)
        return self.solver.hessian()

    def get_shared(self):
        return self.vert

    def vis(self):
        if not self.visualize:
            return
        deformed = self.vert + self.solver.deformation().T  # [n, 3]

        self.mesh.points = deformed
        self.plotter.update_coordinates(deformed, render=False)
        self.plotter.write_frame()

    def step(self):
        self.solver.prepare_step()
        self.solver.update_energy_and_gradients()
        init_force = self.solver.forces().T.copy()
        self.solver.update_hessian_unfiltered()
        hessian = self.solver.hessian()
        self.solver.step()
        curr_deform = self.solver.deformation().T.copy()
        self.vis()

        mask = np.ones((self.nV, self.block_size), dtype=np.int32)
        mask[self.all_bc_idx] = 0

        return hessian, mask, curr_deform, init_force


@hydra.main("config", "elast_bend", "1.3")
def main(cfg: DictConfig):
    logger.info(f"Elasticity config: {cfg}")
    gen = Elasticity(cfg)
    gen.generate()
    if gen.visualize:
        gen.plotter.close()

if __name__ == "__main__":
    main()
