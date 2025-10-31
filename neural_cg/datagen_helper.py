from pathlib import Path
from typing import List, Tuple, Union
from loguru import logger
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import bsr_matrix, csr_matrix, save_npz
import torch
from neural_cg import get_ncg_root
from neural_cg.data import FolderDataset, Data

def load_msh(file_path):
    import meshio
    # Read the mesh file
    mesh = meshio.read(file_path)

    # Extract nodes (vertices)
    nodes = mesh.points  # Shape (nV, 3)

    # Extract tetrahedra (assuming the mesh contains tetrahedral elements)
    # Gmsh stores tetrahedra with cell type "tetra"
    tetra_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetra_cells = cell_block.data
            break

    if tetra_cells is None:
        raise ValueError("No tetrahedral elements found in the mesh.")

    # Note: Some .msh files may use 1-based indexing, so we convert to 0-based if needed
    if tetra_cells.min() == 1:
        tetra_cells -= 1

    return nodes, tetra_cells

def load_obj(file_path):
    import trimesh

    """Load an OBJ file and return vertices and triangular faces.
    
    Args:
        file_path (str): Path to the OBJ file.
    
    Returns:
        vertices (np.ndarray): Vertex coordinates of shape [nV, 3].
        faces (np.ndarray): Triangular face indices of shape [nF, 3] (automatically splits quads).
    """
    mesh = trimesh.load(file_path, force="mesh")

    # Ensure faces are triangular (automatically split quads)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.convex_hull if hasattr(mesh, "convex_hull") else mesh

    # Convert to NumPy arrays
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces

def extract_boundary_faces(tetrahedrons):
    """
    Extract boundary faces from a tetrahedral mesh.

    Args:
        tetrahedrons: (n, 4) array representing the vertex indices of tetrahedrons.

    Returns:
        boundary_faces: A list of boundary faces, each represented as a tuple of vertex indices.
    """
    # 1. Collect all faces and count their occurrences
    face_dict = {}

    for tet in tetrahedrons:
        # Four triangular faces of a tetrahedron
        faces = [
            tuple(sorted((tet[0], tet[1], tet[2]))),
            tuple(sorted((tet[0], tet[1], tet[3]))),
            tuple(sorted((tet[0], tet[2], tet[3]))),
            tuple(sorted((tet[1], tet[2], tet[3]))),
        ]

        for face in faces:
            if face in face_dict:
                face_dict[face] += 1
                assert face_dict[face] <= 2
            else:
                face_dict[face] = 1

    # 2. Identify faces that appear only once (boundary faces)
    boundary_faces = [face for face, count in face_dict.items() if count == 1]

    return boundary_faces

def extract_boundary_vertices(tetrahedrons):
    """
    Extract boundary vertex IDs from a tetrahedral mesh.

    Args:
        tetrahedrons: (n, 4) array representing the vertex indices of tetrahedrons.

    Returns:
        boundary_vertex_ids: A set of boundary vertex IDs.
    """
    boundary_faces = extract_boundary_faces(tetrahedrons)

    # 3. Collect all boundary vertices
    boundary_vertex_ids = set()
    for face in boundary_faces:
        boundary_vertex_ids.update(face)

    return list(boundary_vertex_ids)


def tetrahedralize(v, f, visualize=False, switches="pq1.1/0Y"):
    import pyvista as pv
    from tetgen import TetGen

    tgen = TetGen(v, f)
    tgen.make_manifold()
    nodes, elements = tgen.tetrahedralize(switches=switches)
    if visualize > 0:
        for i in range(1, visualize):
            tet_grid = tgen.grid
            bbox_min = np.min(tet_grid.points, axis=0)
            bbox_max = np.max(tet_grid.points, axis=0)
            ratio = i / visualize
            bbx_center = (1 - ratio) * bbox_min + ratio * bbox_max
            # Plot half the tet.
            mask = tet_grid.points[:, 2] < bbx_center[2]
            half_tet = tet_grid.extract_points(mask)

            plotter = pv.Plotter()
            plotter.add_mesh(half_tet, color="w", show_edges=True)
            plotter.add_mesh(tet_grid, color="r", style="wireframe", opacity=0.2)
            plotter.show()

            plotter.close()
    return nodes.astype(np.float64), elements.astype(np.int32)


def subgraph_grow(
    n_max_count: int,
    seeds: np.ndarray,
    edges: np.ndarray,
    growth_count: int,
) -> np.ndarray:
    """
    Generate subgraphs by growing from seeds with fixed iterations.

    Uses BFS-like propagation for efficient growth tracking.

    Args:
        n_max_count: Maximum node index + 1 (to determine matrix size).
        seeds: 1D array of seed node indices (shape: [n_seeds]).
        edges: 2D array of edges (shape: [n_edges, 2]).
        growth_count: Number of growth iterations from each seed.

    Returns:
        1D array of all nodes reached during growth (including seeds).
    """
    # Create adjacency matrix in CSR format
    adj_matrix = csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_max_count, n_max_count),
    )
    # Make undirected by adding transpose
    adj_matrix = adj_matrix + adj_matrix.T

    # Initialize frontier with seeds (ensure unique)
    frontier = np.unique(seeds)
    visited = np.zeros(n_max_count, dtype=np.float64)
    visited[frontier] = 1.0

    # Grow for specified iterations
    for _ in range(growth_count):
        # Get neighbors of current frontier
        visited = adj_matrix @ visited + visited

    return np.where(visited > 0)[0]

def faces_to_edges(faces: np.ndarray):
    """
    Convert triangular faces to edges.

    Args:
        faces: 2D array of triangular faces (shape: [n_faces, 3]).

    Returns:
        edges: 2D array of edges (shape: [n_edges, 2]).
    """
    edges = np.array(
        [
            [faces[:, 0], faces[:, 1]],
            [faces[:, 1], faces[:, 2]],
            [faces[:, 0], faces[:, 2]],
        ]
    ).reshape(-1, 2)
    return np.unique(np.sort(edges, axis=1), axis=0)

class DatagenBase:
    def __init__(
        self,
        max_count: int,
        block_size: int,
        is_fixed_topology: bool,
        save_lhs: bool,
        save_rhs: bool,
        prefix: str,
        has_shared_features: bool,
        has_node_features: bool,
        dry_run: bool,
    ):
        self.max_count = max_count
        self.block_size = block_size
        self.is_fixed_topology = is_fixed_topology
        self.save_lhs = save_lhs
        self.save_rhs = save_rhs
        self.prefix = prefix
        self.has_shared_features = has_shared_features
        self.has_node_features = has_node_features

        self.current_count = 0
        self.rhs: List[str] = []
        self.lhs: List[str] = []
        self.mat: List[str] = []
        self.path = Path(prefix)
        self.dry_run = dry_run
        if self.dry_run:
            logger.warning("Dry run mode. No data will be saved.")

    def prepare(self):
        logger.info(f"Preparing data generation at {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "mask").mkdir(parents=True, exist_ok=True)
        if self.save_lhs:
            (self.path / "lhs").mkdir(parents=True, exist_ok=True)
        if self.save_rhs:
            (self.path / "rhs").mkdir(parents=True, exist_ok=True)
        (self.path / "mat").mkdir(parents=True, exist_ok=True)
        (self.path / "features").mkdir(parents=True, exist_ok=True)

    def append(
        self,
        mat: csr_matrix,
        mask: Union[np.ndarray, None] = None,
        features: Union[np.ndarray, None] = None,
        rhs: Union[np.ndarray, List[np.ndarray], None] = None,
    ):
        if self.dry_run:
            return mat
        rows = mat.shape[0]  # type: ignore
        # mat.data = mat.data / np.abs(mat.data).mean() # normalize
        # 1. If not fixed topology -> csr save all
        # 2. If fixed ->  values is a vector, in csr
        if self.is_fixed_topology:
            save_mat = mat.sorted_indices()
            np.save(self.path / "mat" / f"{self.current_count:06d}.npy", save_mat.data)
        else:
            # for now. we do not support block_size > 1
            # assert self.block_size == 1
            mmwrite(self.path / "mat" / f"{self.current_count:06d}.mtx", mat)
            # save_npz(self.path / "mat" / f"{self.current_count:06d}.npz", mat)

        ## check & save features
        if features is not None:
            if features.ndim == 1:
                features = features.reshape((-1, 1))
            assert features.shape[0] == rows // self.block_size
            np.save(self.path / "features" / f"{self.current_count:06d}.npy", features)
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.reshape((-1, 1))
            assert mask.shape[0] == rows // self.block_size
            np.save(self.path / "mask" / f"{self.current_count:06d}.npy", mask)

        # if rhs is not provided, lhs does not need to be saved
        if not self.save_rhs:
            return mat

        ## Save vectors.
        if isinstance(rhs, np.ndarray):
            rhs = [rhs.flatten()]
        elif isinstance(rhs, list):
            rhs = [b.flatten() for b in rhs]
        else:
            rhs = []
            cnt = int(self.save_rhs)
            for _ in range(cnt):
                b = np.random.randn(rows)
                b = b / np.linalg.norm(b)
                if mask is not None:
                    b = b * mask.flatten()
                rhs.append(b.flatten().astype(np.float64))

        # if len(rhs) == 0:
        #     if self.save_rhs or self.save_lhs:
        #         logger.error("No vector to save")
        #         exit(1)

        # check rhs's size is capable of the matrix.
        for b in rhs:
            assert b.ndim == 1 and b.size == rows // self.block_size

        ## save rhs
        if self.save_rhs:
            stacked_rhs = np.stack(rhs, axis=0) # [b, nV]
            np.save(self.path / "rhs" / f"{self.current_count:06d}.npy", stacked_rhs.T)

            ## save lhs
            if self.save_lhs:
                from scipy.sparse.linalg import splu
                mat = csr_matrix(mat)
                lhs = []
                decomp = splu(mat)
                for b in rhs:
                    x = decomp.solve(b)
                    lhs.append(x.copy())
                stacked_lhs = np.stack(lhs, axis=0)
                np.save(
                    self.path / "lhs" / f"{self.current_count:06d}.npy",
                    stacked_lhs.T,
                )
        return mat

    def get_shared(self):
        raise NotImplementedError()

    def topology(self) -> csr_matrix:
        raise NotImplementedError()

    def step(
        self,
    ) -> Tuple[
        csr_matrix,
        Union[None, np.ndarray],
        Union[None, np.ndarray],
        Union[None, np.ndarray, List[np.ndarray]],
    ]:
        """
        Returns
            csr_matrix: matrix to solve.
            mask: 1 for free nodes/dofs, 0 for dirichlets
            features: node features
            rhs: right hand side vector
        """
        raise NotImplementedError()

    def generate(self):
        self.prepare()
        if self.is_fixed_topology:
            logger.info("fixed topology")
            topo = self.topology().sorted_indices()
            mmwrite(self.path / "demo.mtx", topo)
        if self.has_shared_features:
            logger.info("shared features")
            np.save(self.path / "shared_features.npy", self.get_shared())

        from tqdm import trange

        with trange(self.max_count) as t:
            while self.current_count < self.max_count:
                mat = self.append(*self.step())
                self.current_count += 1
                t.set_postfix(
                    {
                        "current_count": self.current_count,
                        "max_count": self.max_count,
                        "sparsity": mat.nnz / (mat.shape[0] * mat.shape[1]),
                    }
                )
                t.update(1)

        logger.info("Generation done. check integrity...")
        # check integrity
        dataset = FolderDataset(
            is_fixed_topology=self.is_fixed_topology,
            load_into_memory=False,
            block_size=self.block_size,
            has_shared_features=self.has_shared_features,
            use_node_features=self.has_node_features,
            use_matrix_as_edge_feature=True,
            use_mask_as_node_feature=True,
            use_node_features_as_edge_feature=False,
            use_edge_features_as_node_feature="disable",
            use_random_rhs=int(self.save_rhs) == 0,
            normalize_matrix=False,
            prefix=self.prefix,
        )
        logger.info(f"len of dataset: {len(dataset)}")
        sample: Data = dataset[0]  # try to get one.
        for k, v in sample.to_dict().items():
            if isinstance(v, torch.Tensor):
                logger.info(f" - {k}: {tuple(v.shape)}")

        logger.info("=> Integrity check done.")
        logger.info("All Done.")
