from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Union

from loguru import logger
import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix, diags, load_npz
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import scatter
from collections import defaultdict


def to_bcoo_components(
    coo: coo_matrix,
    block_size: int,
):
    """Convert COO matrix to BCOO components.

    Args:
        coo: Input COO matrix (scipy.sparse.coo_matrix)
        block_size: Size of the blocks (assumed square)

    Returns:
        tuple: (block_values, block_row, block_col) where:
            - block_values: ndarray of shape (NBlock, block_size, block_size)
            - block_row: ndarray of shape (NBlock,) of block row indices
            - block_col: ndarray of shape (NBlock,) of block column indices
    """
    # Input validation
    if not isinstance(coo, coo_matrix):
        raise TypeError("Input must be a scipy.sparse.coo_matrix")
    if block_size <= 0:
        raise ValueError("Block size must be positive")

    rows = coo.row
    cols = coo.col
    data = coo.data

    # Calculate block indices and intra-block positions
    block_rows = rows // block_size
    block_cols = cols // block_size
    intra_rows = rows % block_size
    intra_cols = cols % block_size

    # Dictionary to accumulate blocks: key=(block_row, block_col), value=block matrix
    blocks_dict = defaultdict(lambda: np.zeros((block_size, block_size)))

    # Populate blocks
    for br, bc, ir, ic, val in zip(block_rows, block_cols, intra_rows, intra_cols, data):
        blocks_dict[(br, bc)][ir, ic] = val

    # Convert to output arrays
    NBlock = len(blocks_dict)
    block_values = np.zeros((NBlock, block_size, block_size))
    block_row = np.zeros(NBlock, dtype=int)
    block_col = np.zeros(NBlock, dtype=int)

    for idx, ((br, bc), block) in enumerate(blocks_dict.items()):
        block_values[idx] = block
        block_row[idx] = br
        block_col[idx] = bc

    return block_values, block_row, block_col


def make_bsr_from_csr_inds(
    bsr_values: np.ndarray,
    indptrs: np.ndarray,
    indices: np.ndarray,
    block_size: int,
    block_rows: int,
    block_cols: int,
) -> bsr_matrix:
    """Construct BSR matrix from CSR indices.

    Args:
        bsr_values: The fillin
        indptrs: Row Ptrs
        indices: Col indices
        block_size: Block size
        block_rows: Rows in Block
        block_cols: Cols in Block

    Returns:
        bsr_matrix: shape=(block_rows * block_size, block_cols * block_size)
    """

    assert bsr_values.ndim == 3
    assert indptrs.ndim == 1
    assert indices.ndim == 1
    assert bsr_values.shape[0] == indptrs[-1]
    assert bsr_values.shape[0] == indices.size
    assert bsr_values.shape[1] == bsr_values.shape[2] == block_size
    n = block_rows * block_size
    m = block_cols * block_size
    b = block_size
    return bsr_matrix((bsr_values, indices, indptrs), blocksize=(b, b), shape=(n, m))


# def make_bsr_from_coo_inds(
#     bsr_values: np.ndarray,
#     rowinds: np.ndarray,
#     colinds: np.ndarray,
#     block_size: int,
#     block_rows: int,
#     block_cols: int,
# ) -> bsr_matrix:
#     assert bsr_values.ndim == 3
#     assert rowinds.ndim == 1
#     assert colinds.ndim == 1
#     assert rowinds.size == colinds.size == bsr_values.shape[0]
#     assert bsr_values.shape[1] == bsr_values.shape[2] == block_size
#     n = block_rows * block_size
#     m = block_cols * block_size
#     b = block_size

#     # Repeat rows bs*bs times, and cols bs*bs times
#     rows_repeated = np.repeat(rowinds, block_size * block_size).reshape(-1, block_size, block_size)
#     cols_repeated = np.repeat(colinds, block_size * block_size).reshape(-1, block_size, block_size)

#     for i in range(block_size):
#         rows_repeated[:, i, :] += i
#     for j in range(block_size):
#         cols_repeated[:, :, j] += j
#     return csr_matrix(
#         (bsr_values.flatten(), (rows_repeated.flatten(), cols_repeated.flatten())),
#         shape=(n, m),
#         copy=True,
#     ).sorted_indices()


def make_bsr_from_coo_inds(
    bsr_values: np.ndarray,
    rowinds: np.ndarray,
    colinds: np.ndarray,
    block_size: int,
    block_rows: int,
    block_cols: int,
) -> bsr_matrix:
    assert bsr_values.ndim == 3
    assert rowinds.ndim == 1
    assert colinds.ndim == 1
    assert rowinds.size == colinds.size == bsr_values.shape[0]
    assert bsr_values.shape[1] == bsr_values.shape[2] == block_size
    n = block_rows * block_size
    m = block_cols * block_size
    b = block_size
    csr = csr_matrix(
        (np.ones(rowinds.size), (rowinds, colinds)),
        shape=(block_rows, block_cols),
        copy=True,
    )
    # __import__("pdb").set_trace()
    return bsr_matrix((bsr_values, csr.indices, csr.indptr), blocksize=(b, b), shape=(n, m), copy=True)


def apply_dbc_masking(
    mat: Union[bsr_matrix, csr_matrix],
    mask: np.ndarray,
) -> coo_matrix:
    # first convert to coo format.
    coo = coo_matrix(mat)
    mask_flat = mask.flatten()
    # for those masked, set zeros
    coo.data[mask_flat[coo.row] == 0] = 0
    coo.data[mask_flat[coo.col] == 0] = 0
    ident = (1 - mask_flat).copy()
    return coo + diags(ident, 0, shape=coo.shape)  # type: ignore


@dataclass
class RawData:
    block_values: Union[None, np.ndarray]
    diagonals: Union[None, np.ndarray]
    edge_index: np.ndarray
    node_features: Union[None, np.ndarray]
    lhs: Union[None, np.ndarray]
    rhs: Union[None, np.ndarray]
    mask: np.ndarray
    num_nodes: int
    block_size: int


def _aggregate_edge_to_node(edge_index, edge_attr, num_nodes, reduce="mean"):
    """
    Args:
        edge_index (LongTensor): [2, num_edges]
        edge_attr (Tensor): [num_edges, feature_dim]
        num_nodes (int): Total number of nodes in the graph
        reduce (str): Aggregation method ('sum', 'mean', 'max', 'min', etc.)
    Returns:
        node_features (Tensor): [num_nodes, feature_dim]
    """
    target_nodes = edge_index[1]  # [num_edges]
    node_features = scatter(
        src=edge_attr,  # Input data
        index=target_nodes,  # Target node indices
        dim=0,  # Aggregation dimension
        dim_size=num_nodes,  # Output dimension
        reduce=reduce,  # Aggregation method
    )
    return node_features


def _make_node_from_edge(
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    num_nodes: int,
    use_edge_features_as_node_feature: Literal["disable", "sum", "mean", "max", "min"],
) -> List[torch.Tensor]:
    if use_edge_features_as_node_feature == "disable":
        return []
    return [_aggregate_edge_to_node(edge_index, edge_features, num_nodes, reduce=use_edge_features_as_node_feature)]


def make_data(
    raw_data: RawData,
    use_matrix_as_edge_feature: bool,
    use_mask_as_node_feature: bool,
    use_node_features_as_edge_feature: bool,
    use_edge_features_as_node_feature: Literal["disable", "sum", "mean", "max", "min"],
    use_random_rhs: bool,
    normalize_matrix: bool | str,
    is_inference: bool,
) -> Data:
    """Construct a PyG Data object from raw data with customizable features.

    Args:
        raw_data: Input data container with matrix, graph, and features
        use_matrix_as_edge_feature: Whether to use matrix blocks as edge features
        use_mask_as_node_feature: Whether to include mask in node features
        use_node_features_as_edge_feature: Whether to use node features for edges
        use_edge_features_as_node_feature: How to aggregate edge features to nodes
        use_random_rhs: Whether to generate random RHS vectors
        normalize_matrix: Whether to normalize the matrix
        is_inference: Whether this is inference mode (excludes rhs/lhs)

    Returns:
        Data: PyG graph data with features and optional solution/rhs
    """

    assert not (use_node_features_as_edge_feature and use_edge_features_as_node_feature != "disable")
    edge_index = torch.tensor(raw_data.edge_index, dtype=torch.long)

    matrix_scale: float = 1.0
    if normalize_matrix is True or normalize_matrix == "mean":
        assert raw_data.block_values is not None
        matrix_scale = 1.0 / np.mean(np.abs(raw_data.block_values))  # type: ignore
    elif normalize_matrix == "frob":
        assert raw_data.block_values is not None
        matrix_scale = 1.0 / np.linalg.norm(raw_data.block_values)  # type: ignore
    elif normalize_matrix == "l1":
        assert raw_data.block_values is not None
        bsr = make_bsr_from_coo_inds(
            bsr_values=np.abs(raw_data.block_values),
            rowinds=raw_data.edge_index[0],
            colinds=raw_data.edge_index[1],
            block_size=raw_data.block_size,
            block_rows=raw_data.num_nodes // raw_data.block_size,
            block_cols=raw_data.num_nodes // raw_data.block_size,
        )
        row_sum = bsr @ np.ones(bsr.shape[1])  # type: ignore
        matrix_scale = 1.0 / (np.max(row_sum) + 1e-7)
    elif normalize_matrix == "none" or normalize_matrix is False:
        matrix_scale = 1.0

    # >>> basic node features
    node_features = []
    if raw_data.node_features is not None:
        node_features.append(torch.tensor(raw_data.node_features, dtype=torch.float32))

    n_nodes, bsize = raw_data.num_nodes, raw_data.block_size
    mask = torch.tensor(raw_data.mask, dtype=torch.float32)
    if use_mask_as_node_feature:
        node_features.append(mask)
    # <<< basic node features

    # >>> edge features
    edge_features = []
    if use_matrix_as_edge_feature:
        assert raw_data.block_values is not None
        bvals = torch.tensor(matrix_scale * raw_data.block_values, dtype=torch.float32)
        edge_features.append(bvals.flatten(1))

    if use_node_features_as_edge_feature:
        assert raw_data.node_features is not None
        nf: torch.Tensor = torch.cat(node_features, dim=-1)
        edge_features += [nf[edge_index[i]] for i in [0, 1]]
    assert len(edge_features) > 0, "No edge feature found."
    edge_input = torch.cat(edge_features, dim=-1)  # final edge feature
    # <<< edge features

    # Append a special
    node_features += _make_node_from_edge(
        edge_index,
        edge_input,
        raw_data.num_nodes,
        use_edge_features_as_node_feature,
    )
    assert len(node_features) > 0, "No node feature found."
    node_input = torch.cat(node_features, dim=-1)  # final node feature

    data_dict = {
        "x": node_input,
        "mask": mask,
        "edge_index": edge_index,
        "edge_attr": edge_input,
    }
    if raw_data.block_values is not None:
        data_dict["matrix_values"] = torch.tensor(raw_data.block_values * matrix_scale, dtype=torch.float32)
    else:
        assert not is_inference, "Training depends on matrix values."

    if raw_data.diagonals is not None:
        diag = raw_data.diagonals * matrix_scale
        data_dict["diagonal"] = torch.tensor(diag, dtype=torch.float32)
        # since raw_data is in float64, compute it first is more accurate
        inv_diags = 1.0 / (diag + 1e-7)
        rsqrt_diag = 1.0 / np.sqrt(diag + 1e-7)
        data_dict["inv_diag"] = torch.tensor(inv_diags, dtype=torch.float32)
        data_dict["rsqrt_diag"] = torch.tensor(rsqrt_diag, dtype=torch.float32)

    if not is_inference:
        # TODO: make this a option (randn)
        rhs = torch.randn(n_nodes, bsize, dtype=torch.float32)
        if not use_random_rhs:
            assert raw_data.rhs is not None
            rhs = torch.tensor(raw_data.rhs, dtype=torch.float32)
            if raw_data.lhs is not None:
                # gt is provided, set it in the data.
                data_dict["gt"] = torch.tensor(raw_data.lhs, dtype=torch.float32) / matrix_scale
        rhs = rhs * mask
        data_dict["residual"] = rhs
    return Data(**data_dict)


class FolderDataset(Dataset):
    def __init__(
        self,
        is_fixed_topology: bool,
        load_into_memory: bool,
        block_size: int,
        has_shared_features: bool,
        use_node_features: bool,
        use_matrix_as_edge_feature: bool,
        use_mask_as_node_feature: bool,
        use_node_features_as_edge_feature: bool,
        use_edge_features_as_node_feature: Literal["disable", "sum", "mean", "max", "min"],
        use_random_rhs: bool,
        normalize_matrix: bool | str,
        prefix: str,
    ):
        super().__init__()
        self.is_fixed_topology = is_fixed_topology
        self.prefix = prefix
        self.block_size = block_size
        self.path = Path(prefix)

        if self.is_fixed_topology:
            all_matrices = list((self.path / "mat").glob("*.npy"))
        else:
            all_matrices = list((self.path / "mat").glob("*.mtx")) + list((self.path / "mat").glob("*.npz"))
        all_lhs = list((self.path / "lhs").glob("*.npy"))
        all_rhs = list((self.path / "rhs").glob("*.npy"))
        all_masks = list(self.path.glob("mask/*.npy"))
        all_features = list((self.path / "features").glob("*.npy"))
        all_matrices.sort()
        all_lhs.sort()
        all_rhs.sort()
        all_masks.sort()
        all_features.sort()

        logger.info(
            f"Found {len(all_matrices)} matrices, {len(all_lhs)} lhs, {len(all_rhs)} rhs, {len(all_masks)} masks, {len(all_features)} features"
        )

        self.has_shared_features = has_shared_features
        if self.has_shared_features:
            self.shared_features = np.load(self.path / "shared_features.npy")
        else:
            self.shared_features = None

        assert len(all_matrices) > 0
        if all_lhs:
            assert len(all_lhs) == len(all_matrices)
        if all_rhs:
            assert len(all_rhs) == len(all_matrices)

        # Ax = b
        samples: List[Tuple[int, int]] = []
        for idx, f in enumerate(all_rhs):
            b = np.load(f)
            for i in range(b.shape[1]):
                samples.append((idx, i))
        self.samples = samples

        self.all_matrices = all_matrices
        self.all_lhs = all_lhs
        self.all_rhs = all_rhs
        self.all_features = all_features
        self.all_masks = all_masks

        self.use_node_features = use_node_features
        self.use_matrix_as_edge_feature = use_matrix_as_edge_feature
        self.use_mask_as_node_feature = use_mask_as_node_feature
        self.use_node_features_as_edge_feature = use_node_features_as_edge_feature
        self.use_edge_features_as_node_feature = use_edge_features_as_node_feature
        self.use_random_rhs = use_random_rhs

        # >>> node/edge feature count
        self.num_node_features_ = 0  # avoid the naming conflict
        if self.use_node_features:
            assert len(self.all_features) == len(self.all_matrices)
            self.num_node_features_ = np.load(self.all_features[0]).shape[1]
            if self.has_shared_features:
                self.num_node_features_ += self.shared_features.shape[1]  # type: ignore
        if self.use_mask_as_node_feature:
            self.num_node_features_ += self.block_size

        if self.use_node_features_as_edge_feature and self.use_edge_features_as_node_feature != "disable":
            raise ValueError("You cannot enable both feature enhancers")

        self.num_edge_features_ = 0
        if self.use_matrix_as_edge_feature:
            self.num_edge_features_ += self.block_size * self.block_size
        if self.use_node_features_as_edge_feature:
            self.num_edge_features_ += self.num_node_features_ * 2
        if self.use_edge_features_as_node_feature != "disable":
            self.num_node_features_ += self.num_edge_features_
        # <<< node/edge feature count

        self.normalize_matrix: bool | str = normalize_matrix
        # The topology file is BSR's idx, also the graph
        if self.is_fixed_topology:
            self.topo_file = self.path / "demo.mtx"
            assert self.topo_file.exists()
            self.topo_mat_dofs = csr_matrix(mmread(self.topo_file)).sorted_indices()
            topo_mat_graph = bsr_matrix(self.topo_mat_dofs.tobsr((self.block_size, self.block_size))).sorted_indices()
            num_nodes = topo_mat_graph.shape[0] // self.block_size  # type: ignore
            self.topo_mat_graph = coo_matrix(
                csr_matrix(
                    (np.ones(topo_mat_graph.indptr[-1]), topo_mat_graph.indices, topo_mat_graph.indptr),  # type: ignore
                    shape=(num_nodes, num_nodes),
                )
            )
            self.rowinds = self.topo_mat_graph.row
            self.colinds = self.topo_mat_graph.col
            self.edge_index = np.vstack((self.rowinds, self.colinds)).astype(np.int64)
            logger.info(f"Fixed Topology, matrix shape={self.topo_mat_graph.shape}, nnz={self.topo_mat_graph.nnz}")
        logger.info(f"Dataset folder: {prefix}")
        self.loaded = []
        if load_into_memory:
            from tqdm import trange

            loaded = []
            for i in trange(len(self), desc=f"Load Folder {prefix}"):
                loaded.append(self.get_internal(i))
            self.loaded = loaded
            for k, v in self.get(0).to_dict().items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"{k}: {v.shape}")
            logger.info(f"Loaded {len(self.loaded)} samples into memory.")

    def len(self) -> int:
        len_mat = len(self.all_matrices)
        len_lhs = len(self.samples)
        return max(len_mat, len_lhs)

    def load(
        self,
        mat_file: Path,
        lhs_file: Union[None, Path],
        rhs_file: Union[None, Path],
        feature_file: Union[None, Path],
        mask_file: Union[None, Path],
        dtype=np.float64,
    ):
        if str(mat_file).endswith(".npy"):
            # only values are stored.
            assert self.is_fixed_topology
            values = np.load(mat_file)
            edge_index = self.edge_index
            num_nodes = self.topo_mat_graph.shape[0]  # type: ignore
            matrix_csr = csr_matrix(
                (values, self.topo_mat_dofs.indices, self.topo_mat_dofs.indptr),  # type: ignore
                shape=self.topo_mat_dofs.shape,
            )
            matrix = bsr_matrix(matrix_csr.tobsr((self.block_size, self.block_size))).sorted_indices()
            block_values = matrix.data.astype(dtype).copy()
        else:  # is a full matrix file
            # assert self.block_size == 1
            if str(mat_file).endswith(".mtx"):
                matrix = csr_matrix(mmread(mat_file))
            else:
                assert False  # It seems scipy's npz don't work properly?
                assert str(mat_file).endswith(".npz")
                matrix = csr_matrix(load_npz(mat_file))
            coo = coo_matrix(matrix)
            if self.block_size == 1:
                block_values = coo.data.astype(dtype).reshape(-1, 1, 1).copy()
                edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)  # type: ignore
                num_nodes = matrix.shape[0]  # type: ignore
            else:
                block_values, brows, bcols = to_bcoo_components(coo, self.block_size)
                edge_index = np.vstack((brows, bcols)).astype(np.int64)  # type: ignore
                num_nodes = matrix.shape[0] // self.block_size  # type: ignore

        diagonals: np.ndarray = matrix.diagonal().reshape(-1, self.block_size)
        lhs = None
        rhs = None
        node_features = None

        if rhs_file is not None:
            rhs = np.load(rhs_file)
            if rhs.ndim == 1:
                rhs = rhs.reshape(-1, 1)
            elif rhs.ndim > 2:
                raise ValueError(f"Unexpected RHS shape: {rhs.shape}")
            assert rhs.shape[0] == num_nodes

            if lhs_file is not None:
                lhs = np.load(lhs_file)
                if lhs.ndim == 1:
                    lhs = lhs.reshape(-1, 1)
                elif lhs.ndim > 2:
                    raise ValueError(f"Unexpected LHS shape: {lhs.shape}")
                assert lhs.shape == rhs.shape

        if self.use_node_features:
            node_feat_list = []
            if feature_file is not None:
                feat = np.load(feature_file)
                assert feat.ndim == 2
                assert feat.shape[0] == num_nodes
                node_feat_list.append(feat)
            if self.has_shared_features:
                node_feat_list.append(self.shared_features)
            node_features = np.concatenate(node_feat_list, axis=-1)

        mask = np.ones((num_nodes, self.block_size), dtype=dtype)
        if mask_file is not None:
            mask_loaded = np.load(mask_file)
            assert mask_loaded.shape == mask.shape
            mask = mask_loaded

        raw = RawData(
            block_values,
            diagonals,
            edge_index,
            node_features,
            lhs,
            rhs,
            mask,
            num_nodes,
            self.block_size,
        )

        # XXX: remove this in the future, this code checks the lhs, rhs is correct
        # from neural_cg.utils.validate import to_csr_cpu, to_numpy
        # gt = lhs[:, 0]
        # residual = rhs[:, 0]
        # A = to_csr_cpu(raw.edge_index, raw.block_values, raw.num_nodes, raw.mask, dtype=np.float64).tocoo()
        # print('row diff: ', np.abs(A.row - matrix.row).sum())
        # print('col diff: ', np.abs(A.col - matrix.col).sum())
        # print('data diff: ', np.abs(A.data - matrix.data).mean())
        # print(np.linalg.norm(matrix @ lhs - rhs), np.linalg.norm(A @ lhs - rhs))
        # logger.info(f"|r| = {np.linalg.norm(A @ lhs.copy() - rhs.copy())}")
        # logger.info(f"|r| = {np.linalg.norm(A @ gt.copy() - residual.copy())}")

        return raw

    @property
    def num_node_features(self) -> int:
        return self.num_node_features_

    @property
    def num_edge_features(self) -> int:
        return self.num_edge_features_

    def get_internal(self, idx: int) -> RawData:
        if self.loaded:
            return self.loaded[idx]
        if self.samples:
            mat_id, sub_id = self.samples[idx]
        else:
            mat_id = idx
            sub_id = 0

        matrix = self.all_matrices[mat_id]
        lhs = self.all_lhs[mat_id] if self.all_lhs else None
        rhs = self.all_rhs[mat_id] if self.all_rhs else None
        feature = self.all_features[mat_id] if self.all_features else None
        masks = self.all_masks[mat_id] if self.all_masks else None

        raw = self.load(
            mat_file=matrix,
            lhs_file=lhs,
            rhs_file=rhs,
            feature_file=feature,
            mask_file=masks,
        )

        raw_sub = RawData(
            block_values=raw.block_values,
            diagonals=raw.diagonals,
            edge_index=raw.edge_index,
            node_features=raw.node_features,
            lhs=raw.lhs[:, sub_id].reshape(-1, self.block_size) if raw.lhs is not None else None,
            rhs=raw.rhs[:, sub_id].reshape(-1, self.block_size) if raw.rhs is not None else None,
            mask=raw.mask,
            num_nodes=raw.num_nodes,
            block_size=raw.block_size,
        )

        # # XXX: remove this in the future, this code checks the lhs, rhs is correct
        # from neural_cg.utils.validate import to_csr_cpu, to_numpy
        # residual = raw_sub.rhs.flatten()
        # gt = raw_sub.lhs.flatten()
        # A = to_csr_cpu(raw.edge_index, raw.block_values, residual.shape[0], raw.mask, dtype=np.float64)
        # r = A @ to_numpy(gt).flatten() - to_numpy(residual).flatten()
        # logger.info(f"|r|/|rhs| = {np.linalg.norm(r) / np.linalg.norm(residual.flatten())}")
        return raw_sub

    def get(self, idx: int):
        data = make_data(
            self.get_internal(idx),
            self.use_matrix_as_edge_feature,
            self.use_mask_as_node_feature,
            self.use_node_features_as_edge_feature,
            self.use_edge_features_as_node_feature,  # type: ignore
            self.use_random_rhs,
            self.normalize_matrix,
            is_inference=False,
        )

        assert data.x.shape[-1] == self.num_node_features  # type: ignore
        assert data.edge_attr.shape[-1] == self.num_edge_features  # type: ignore
        return data


class MultiFolderDataset(Dataset):
    def __init__(self, all_prefix: List[str], *args, **kwargs):
        super().__init__()
        if "prefix" in kwargs:
            kwargs.pop("prefix")
        self.datasets = []
        for prefix in all_prefix:
            self.datasets.append(FolderDataset(prefix=prefix, *args, **kwargs))
        self.dataset_length = len(self.datasets[0])
        self.block_size = self.datasets[0].block_size

    def get(self, idx):
        dataset_idx, data_idx = idx // self.dataset_length, idx % self.dataset_length
        return self.datasets[dataset_idx].get(data_idx)

    @property
    def num_edge_features(self) -> int:
        return self.datasets[0].num_edge_features

    @property
    def num_node_features(self) -> int:
        return self.datasets[0].num_node_features

    def len(self):
        return self.dataset_length * len(self.datasets)

