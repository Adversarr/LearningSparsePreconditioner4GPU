import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyssim.fem import TetFiniteElementSolver_Host, unit_box
import pyvista as pv
from neural_cg.datagen_helper import DatagenBase, load_obj, tetrahedralize
from matplotlib import pyplot as plt


class Elasticity(DatagenBase):
    def __init__(self, config):
        super().__init__(**config.basic)
        vert, face = load_obj(config.mesh_path)
        logger.info(
            f"Loaded {config.mesh_path}, vert: {vert.shape}, face: {face.shape}"
        )
        vert_e, elem_e = tetrahedralize(vert, face)
        logger.info(
            f"tetrahedralized {config.mesh_path}, vert: {vert_e.shape}, elem: {elem_e.shape}"
        )

        self.vert = vert_e
        self.elem = elem_e
        self.time_step = config.time_step
        self.density = config.density
        self.youngs = config.youngs
        self.poisson = config.poisson

        tsolver = TetFiniteElementSolver_Host(
            self.vert,
            self.elem,
            method="lbfgs_pd",
            time_step=self.time_step,
            young_modulus=self.youngs,
            poisson_ratio=self.poisson,
            density=self.density,
        )

        def in_ball(x, y, z, r):
            return (vert[:, 0] - x) ** 2 + (vert[:, 1] - y) ** 2 + (
                vert[:, 2] - z
            ) ** 2 < r**2

        tsolver.set_rtol(1e-4)
        left_bc = in_ball(-0.36, 0.31, 0.32, 0.04)
        self.left_hand = left_bc_dofs = np.where(left_bc)[0]
        left_bc_deform = np.zeros((left_bc_dofs.size, 3), dtype=np.float64).T
        right_bc = in_ball(0.36, 0.22, 0.38, 0.04)
        tsolver.mark_dirichlet_batched(left_bc_dofs, left_bc_deform)
        self.right_hand = right_bc_dofs = np.where(right_bc)[0]
        right_bc_deform = np.zeros((right_bc_dofs.size, 3), dtype=np.float64).T
        tsolver.mark_dirichlet_batched(right_bc_dofs, right_bc_deform)

        self.left_foot = left_bc_dofs = np.where(in_ball(-0.36, -0.01, -0.47, 0.04))[0]
        left_bc_deform = np.zeros((left_bc_dofs.size, 3), dtype=np.float64).T
        tsolver.mark_dirichlet_batched(left_bc_dofs, left_bc_deform)
        self.right_foot = right_bc_dofs = np.where(in_ball(0.20, -0.01, -0.47, 0.04))[0]
        right_bc_deform = np.zeros((right_bc_dofs.size, 3), dtype=np.float64).T
        tsolver.mark_dirichlet_batched(right_bc_dofs, right_bc_deform)

        tsolver.reset()
        self.nV = self.vert.shape[0]
        self.nE = self.elem.shape[0]
        self.all_bc_idx = np.concatenate([left_bc_dofs, right_bc_dofs])
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
            self.plotter.camera.focal_point = [0, 0, 0]
            self.plotter.camera.position = [0, 5, 0]
            self.plotter.add_mesh(self.mesh, show_edges=True)
            self.plotter.open_gif("armadillo.gif", fps=100)
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
        # self.plotter.render()

    def step(self):
        mask = np.ones((self.nV, self.block_size), dtype=np.float32)
        if self.current_count < 500:
            mask[self.all_bc_idx] = 0
            if self.current_count < 300:
                # stretch left along [-1, 0, 1]
                dx = self.T * np.ones((3, self.left_hand.shape[0]), dtype=np.float64) * 0.5
                dx[0] *= -1
                dx[1] = 0
                self.solver.mark_dirichlet_batched(self.left_hand, dx)
                # stretch right foot along [1, 0, -1]
                dx = self.T * np.ones((3, self.right_foot.shape[0]), dtype=np.float64) * 0.5
                dx[1] = 0
                dx[2] *= -1
                self.solver.mark_dirichlet_batched(self.right_foot, dx)
            else:
                self.solver.mark_general_batched(self.left_hand)
                self.solver.mark_general_batched(self.right_foot)
                mask[self.left_hand] = 1
                mask[self.right_foot] = 1

            # stretch right along [1, 0, 1]
            dx = self.T * np.ones((3, self.right_hand.shape[0]), dtype=np.float64) * 0.5
            dx[1] = 0
            self.solver.mark_dirichlet_batched(self.right_hand, dx)

            # stretch left foot along [-1, 0, -1]
            dx = self.T * np.ones((3, self.left_foot.shape[0]), dtype=np.float64) * 0.5
            dx[0] *= -1
            dx[1] = 0
            dx[2] *= -1
            self.solver.mark_dirichlet_batched(self.left_foot, dx)
        else:
            self.solver.mark_general_batched(self.left_hand)
            self.solver.mark_general_batched(self.right_hand)
            self.solver.mark_general_batched(self.left_foot)
            self.solver.mark_general_batched(self.right_foot)

        self.solver.prepare_step()
        self.solver.update_energy_and_gradients()
        init_force = self.solver.forces().T.copy()
        self.solver.update_hessian_unfiltered()
        hessian = self.solver.hessian()
        self.solver.step()
        curr_deform = self.solver.deformation().T.copy() # shape= [nV, 3]
        self.T += self.time_step
        self.vis()

        # make curr_deform zero centered.
        curr_deform -= np.mean(curr_deform, axis=0, keepdims=True)
        return hessian, mask, curr_deform, init_force


@hydra.main("config", "stretch_armadillo", "1.3")
def main(cfg: DictConfig):
    logger.info(f"Elasticity config: {cfg}")
    gen = Elasticity(cfg)
    gen.generate()
    if gen.visualize:
        gen.plotter.close()


if __name__ == "__main__":
    main()
