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
    def __init__(self, config: DictConfig):
        super().__init__(**config.basic)
        resolution: int = config.resolution
        self.nx = nx = config.nx
        vert, elem = unit_box(resolution * nx, resolution, resolution)
        vert = vert.T.copy()
        elem = elem.T.copy()
        vert[:, 0] *= nx
        vert[:, 1] -= 0.5
        vert[:, 2] -= 0.5

        if 'mesh' in config:
            prefix = config.mesh
            vert = np.load(prefix + "_verts.npy")
            elem = np.load(prefix + "_elems.npy")
            self.vert = vert
            self.elem = elem
            self.nV = vert.shape[0]
            logger.warning(f"Loaded mesh from file: {prefix}, shape={vert.shape}, {elem.shape}")
        else:
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

        tsolver.set_rtol(3e-4)
        tsolver.add_gravity(np.array([0, 0, -9.8], dtype=np.float64))
        left_bc = vert[:, 0] == vert[:, 0].min()
        self.left_bc_dofs = left_bc_dofs = np.where(left_bc)[0]
        left_bc_deform = np.zeros((left_bc_dofs.size, 3), dtype=np.float64).T
        right_bc = vert[:, 0] == vert[:, 0].max()
        self.right_bc_dofs = right_bc_dofs = np.where(right_bc)[0]
        tsolver.mark_dirichlet_batched(left_bc_dofs, left_bc_deform)
        right_bc_deform = left_bc_deform.copy()
        tsolver.mark_dirichlet_batched(right_bc_dofs, right_bc_deform)
        tsolver.reset()
        self.nV = vert.shape[0]
        self.nE = elem.shape[0]
        self.all_bc_idx = np.concatenate([left_bc_dofs, right_bc_dofs])
        logger.info(f"Length of bc = {len(self.all_bc_idx)}")

        self.solver = tsolver
        self.vert_right_original = vert[right_bc_dofs].copy()
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
            self.plotter.camera.position = [self.nx, self.nx + 5, 2]
            self.plotter.add_mesh(self.mesh, show_edges=True)
            self.plotter.show(interactive=False, auto_close=False, window_size=(3840, 2160))

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
        self.plotter.update_coordinates(deformed)
        self.plotter.render()
        self.plotter.show(interactive=False, auto_close=False, window_size=(3840, 2160), screenshot=True)
        self.plotter.screenshot(f"deform_{self.current_count == 0}.png")  # Saves to PNG file

    def step(self):
        vert_right_curr = self.vert_right_original @ rotate_around_x(self.T)
        self.T += self.time_step
        vert_right_deform = vert_right_curr - self.vert_right_original
        self.solver.mark_dirichlet_batched(self.right_bc_dofs, vert_right_deform.T)
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


@hydra.main("config", "elast_twist", "1.3")
def main(cfg: DictConfig):
    logger.info(f"Elasticity config: {cfg}")
    gen = Elasticity(cfg)
    gen.generate()
    if gen.visualize:
        gen.plotter.close()

if __name__ == "__main__":
    main()
