import hydra
from loguru import logger
import numpy as np
from pymathprim.geometry import laplacian
from scipy.sparse import csr_matrix

from neural_cg.data import apply_dbc_masking
from neural_cg.datagen_helper import DatagenBase, load_obj, tetrahedralize

def extract_boundary_vertices(tetrahedrons):
    """
    Extract boundary vertex IDs from a tetrahedral mesh.

    Args:
        tetrahedrons: (n, 3) array representing the vertex indices of tetrahedrons.

    Returns:
        boundary_vertex_ids: A set of boundary vertex IDs.
    """
    # 1. Collect all faces and count their occurrences
    face_dict = {}

    for tet in tetrahedrons:
        # Four triangular faces of a tetrahedron
        lines = [
            tuple(sorted((tet[0], tet[1],))),
            tuple(sorted((tet[0], tet[2],))),
            tuple(sorted((tet[1], tet[2],))),
        ]

        for line in lines:
            if line in face_dict:
                face_dict[line] += 1
                assert face_dict[line] <= 2
            else:
                face_dict[line] = 1

    # 2. Identify faces that appear only once (boundary faces)
    boundary_faces = [face for face, count in face_dict.items() if count == 1]

    # 3. Collect all boundary vertices
    boundary_vertex_ids = set()
    for line in boundary_faces:
        boundary_vertex_ids.update(line)

    return list(boundary_vertex_ids)

class PoissonDatagen(DatagenBase):
    def __init__(self, config):
        super().__init__(**config.basic)
        mesh_file = config.mesh_file
        self.nodes, self.elem = load_obj(mesh_file)
        self.nodes = self.nodes.astype(np.float64)
        self.elem = self.elem.astype(np.int32)
        logger.info(f"Manifold nodes: {self.nodes.shape[0]}")
        logger.info(f"Manifold faces: {self.elem.shape[0]}")
        n_nodes = self.nodes.shape[0]
        n_elems = self.elem.shape[0]
        physics = self.nodes.shape[1]
        intrisic = self.elem.shape[1] - 1
        logger.info(
            f"Heta Dataset for Single Mesh: {mesh_file} => n_nodes: {n_nodes}, n_elems: {n_elems}, "
            f"intrisic: {intrisic}, physics: {physics}"
        )
        self.boundaries = np.array(extract_boundary_vertices(self.elem))
        self.ratio = config.ratio
        assert len(self.boundaries) > 1
        logger.info(f"Boundary shape: {self.boundaries.shape}")

    def get_shared(self):
        return self.nodes

    def topology(self) -> csr_matrix:
        return laplacian(self.nodes, self.elem).sorted_indices()

    def step(self):
        L = laplacian(self.nodes, self.elem)
        dbc_cnt = int(self.ratio * len(self.boundaries))
        mask = np.ones(L.shape[0], dtype=np.float64).reshape(-1, 1)
        dbc = np.random.choice(self.boundaries.shape[0], size=dbc_cnt, replace=False)
        mask[self.boundaries[dbc]] = 0
        # Filter out dbc.
        L = apply_dbc_masking(L, mask=mask)
        return L, mask, None, None

@hydra.main(config_path="config", config_name="poisson", version_base="1.3")
def main(cfg):
    print(cfg)
    datagen = PoissonDatagen(cfg)
    datagen.generate()

if __name__ == "__main__":
    main()