import time
from typing import Tuple, Union, override

import numpy as np
from pandas import timedelta_range
from scipy.sparse import bsr_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, spsolve_triangular
import torch

from ..data import apply_dbc_masking, make_bsr_from_coo_inds


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Unknown type {type(x)}")


def to_csr_cpu(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n: int,
    mask: Union[None, torch.Tensor],
    dtype: np.dtype = np.float64,  # type: ignore
) -> csr_matrix:
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_attr.ndim in [1, 3]  # csr or bsr
    row, col = edge_index
    bsize = edge_attr.shape[-1]
    row_np = to_numpy(row).astype(np.int32)
    col_np = to_numpy(col).astype(np.int32)
    vals_original = to_numpy(edge_attr).astype(dtype)
    if vals_original.shape[1] > 1:
        mat = make_bsr_from_coo_inds(
            bsr_values=vals_original,
            rowinds=row_np,
            colinds=col_np,
            block_size=bsize,
            block_rows=n // bsize,
            block_cols=n // bsize,
        )
    else:
        mat = vals_original.flatten()
        mat = csr_matrix((mat, (row_np, col_np)), shape=(n, n), dtype=dtype)
    if mask is not None:
        mask_np = to_numpy(mask).flatten().astype(dtype)
        mat = apply_dbc_masking(mat, mask=mask_np)
    return csr_matrix(mat).sorted_indices()


def get_cg_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    repeat=1,
    device="cpu",
    method="ainv",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    iter_cnt = 0
    time_prec = 0
    time_elp = 0
    b = b.astype(dtype).copy()
    A = A.astype(dtype)

    from pymathprim.linalg import PreconditionedConjugateGradient as pcg

    x: np.ndarray = np.zeros_like(b, dtype=dtype)
    for _ in range(repeat):
        x_copy = x.copy()
        b_copy = b.copy()
        solver = pcg(matrix=A, device=device, preconditioner=method, dtype=np.float64)  # type: ignore
        this_iter, this_prec, this_solve = solver(b_copy, x_copy, rtol, max_iter)
        iter_cnt += this_iter
        time_prec += this_prec
        time_elp += this_solve
        if this_iter >= max_iter:
            raise RuntimeError("CG did not converge")
    return iter_cnt / repeat, time_prec / repeat, time_elp / repeat


def get_pcg_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    spai: csr_matrix,
    epsilon: float,
    rtol=1e-6,
    max_iter=0,
    repeat=1,
    dtype=np.float64,
    device="cpu",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    assert repeat > 0
    iter_cnt = 0
    time_elp = 0
    time_prec = 0

    A = A.astype(dtype)
    spai = spai.astype(dtype)
    from pymathprim.linalg import PreconditionedConjugateGradient as pcg

    x: np.ndarray = np.zeros_like(b, dtype=dtype)
    for _ in range(repeat):
        x_copy = x.copy()
        b_copy = b.copy()
        solver = pcg(A, device=device, preconditioner="ext_spai", dtype=np.float64)  # type: ignore
        this_iter, this_prec, this_solve = solver(b_copy, x_copy, rtol, max_iter, ext_spai=(spai, epsilon))
        iter_cnt += this_iter
        time_prec += this_prec
        time_elp += this_solve
    return iter_cnt / repeat, time_prec / repeat, time_elp / repeat


def get_pcg_scaled_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    spai: csr_matrix,
    epsilon: float,
    rtol=1e-6,
    max_iter=0,
    repeat=1,
    dtype=np.float64,
    device="cpu",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    assert repeat > 0
    iter_cnt = 0
    time_elp = 0
    time_prec = 0

    A = A.astype(dtype)
    spai = spai.astype(dtype)
    from pymathprim.linalg import PreconditionedConjugateGradient as pcg

    x: np.ndarray = np.zeros_like(b, dtype=dtype)
    for _ in range(repeat):
        x_copy = x.copy()
        b_copy = b.copy()
        solver = pcg(
            A,
            device=device,  # type: ignore
            preconditioner="ext_spai_scaled",
        )
        this_iter, this_prec, this_solve = solver(b_copy, x_copy, rtol, max_iter, ext_spai=(spai, epsilon))
        iter_cnt += this_iter
        time_prec += this_prec
        time_elp += this_solve
    return iter_cnt / repeat, time_prec / repeat, time_elp / repeat


def get_pcg_iter_time_scipy(
    A: csr_matrix,
    gt: np.ndarray,
    spai: csr_matrix,
    epsilon: float,
    max_iter=0,
    rtol=1e-6,
    dtype=np.float64,
    with_time: bool = False,
):
    class Precond(LinearOperator):
        def __init__(self, spai: csr_matrix, epsilon):
            self.spai = spai
            self.trans_spai = csr_matrix(spai.T)
            self.epsilon = epsilon
            super().__init__(self.spai.dtype, spai.shape)

        @override
        def _matvec(self, x):
            return self.spai @ (self.trans_spai @ x) + self.epsilon * x

    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    A = A.astype(dtype)
    spai = spai.astype(dtype)

    M = Precond(spai, epsilon)
    b: np.ndarray = A @ gt
    counter = 0

    def counter_callback(x):
        nonlocal counter
        counter += 1
    time_start = time.time()
    _, _ = cg(A, b, M=M, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    time_end = time.time()
    if with_time:
        return counter, time_end - time_start
    return counter



def get_pcg_ic_iter_time_scipy(
    A: csr_matrix,
    gt: np.ndarray,
    max_iter=0,
    dtype=np.float64,
    rtol=1e-6,
    with_time: bool = False,
):
    from ilupp import IChol0Preconditioner
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    A = A.astype(dtype)
    M = IChol0Preconditioner(A)
    b: np.ndarray = A @ gt

    counter = 0

    def counter_callback(x):
        nonlocal counter
        counter += 1

    time_beg = time.time()
    _, _ = cg(A, b, M=M, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    time_end = time.time()
    if with_time:
        return counter, time_end - time_beg
    return counter



def get_pcg_diagonal_iter_time_scipy(
    A: csr_matrix,
    gt: np.ndarray,
    max_iter=0,
    rtol=1e-6,
    dtype=np.float64,
):
    class Precond(LinearOperator):
        def __init__(self, A: csr_matrix):
            self.diags = A.diagonal()
            super().__init__(self.diags.dtype, A.shape)

        @override
        def _matvec(self, x):
            return x / self.diags

    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    A = A.astype(dtype)
    M = Precond(A)
    b: np.ndarray = A @ gt

    counter = 0

    def counter_callback(x):
        nonlocal counter
        counter += 1

    _, _ = cg(A, b, M=M, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    return counter


def get_pcg_scaled_iter_time_scipy(
    A: csr_matrix,
    gt: np.ndarray,
    spai: csr_matrix,
    epsilon: float,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
):
    class Precond(LinearOperator):
        def __init__(self, A: csr_matrix, spai: csr_matrix, epsilon):
            self.spai = spai
            self.trans_spai = csr_matrix(spai.T)
            self.epsilon = epsilon
            self.diags = A.diagonal()
            super().__init__(self.spai.dtype, spai.shape)

        @override
        def _matvec(self, x):
            return self.spai @ ((self.trans_spai @ x) / self.diags) + self.epsilon * x / self.diags

    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    A = A.astype(dtype)
    spai = spai.astype(dtype)
    M = Precond(A, spai, epsilon)
    b: np.ndarray = A @ gt

    counter = 0

    def counter_callback(x):
        nonlocal counter
        counter += 1

    _, _ = cg(A, b, M=M, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    return counter

def get_pcg_stat_scipy(
    A: csr_matrix,
    r: np.ndarray,
    prefix: str
): 
    stats = {}
    stats[f"{prefix}/cpu_none_iter"] = get_cg_iter_time_scipy(A, r)
    stats[f"{prefix}/cpu_diag_iter"] = get_pcg_diagonal_iter_time_scipy(A, r)
    stats[f"{prefix}/cpu_ic0_iter"] = get_pcg_ic_iter_time_scipy(A, r)
    return stats


def get_cg_iter_time_scipy(
    A: csr_matrix,
    gt: np.ndarray,
    max_iter=0,
    rtol=1e-6,
    dtype=np.float64,
):
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    A = A.astype(dtype)
    b: np.ndarray = A @ gt

    counter = 0
    def counter_callback(x):
        nonlocal counter
        counter += 1
    _, _ = cg(A, b, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    return counter
    counter = 0

    def counter_callback(x):
        nonlocal counter
        counter += 1

    _, _ = cg(A, b, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    return counter


class IncompleteCholeskyPreconditioner(LinearOperator):
    def __init__(self, L):
        """
        Incomplete Cholesky preconditioner using a lower triangular matrix L.

        Parameters
        ----------
        L : array_like or sparse matrix
            Lower triangular matrix from incomplete Cholesky factorization
        """
        self.L = csc_matrix(L)
        self.Lt = csc_matrix(L.T)
        self.shape = L.shape
        self.dtype = L.dtype

    def _matvec(self, x):
        """Apply the preconditioner: solve L L^T x = b"""
        # Solve L y = x
        y = spsolve_triangular(self.L, x, overwrite_A=True, lower=True)
        # Solve L^T z = y
        z = spsolve_triangular(self.Lt, y, overwrite_A=True,  lower=False)
        return z

    def _rmatvec(self, x):
        """Apply the transposed preconditioner (same as matvec for symmetric case)"""
        return self._matvec(x)


def get_pcg_iter_time_scipy_ichol(
    A: csr_matrix,
    L: csr_matrix,
    gt,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    with_time: bool = False,
):
    """
    Get the number of iterations for the preconditioned conjugate gradient method
    using an incomplete Cholesky preconditioner.
    Parameters
    ----------
    A : csr_matrix
        The matrix to solve.
    L : csr_matrix
        The incomplete Cholesky factorization of A.
    gt : np.ndarray
        The right-hand side vector.
    max_iter : int
        The maximum number of iterations.
    dtype : np.dtype
        The data type of the matrix and vector.
    Returns
    -------
    int
        The number of iterations.
    """

    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = A @ gt
    L = L.astype(dtype)
    A = A.astype(dtype)

    counter = 0

    def counter_callback(x):
        nonlocal counter
        counter += 1
    time_beg = time.time()
    ic = IncompleteCholeskyPreconditioner(L)
    _, _ = cg(A, b, M=ic, callback=counter_callback, rtol=rtol, maxiter=max_iter)
    time_end = time.time()
    if with_time:
        return counter, time_end - time_beg
    return counter


# if __name__ == "__main__":
#     from ilupp import IChol0Preconditioner
#     import scipy.sparse as sp

#     A = sp.random(1000, 1000, density=0.001, format="csc")
#     A = A @ A.T + sp.eye(1000) * 1e-6
#     lu = IChol0Preconditioner(A)
#     L = lu.factors()[0]

#     gt = np.random.rand(1000)
#     print(get_pcg_iter_time_scipy_ichol(A, L, gt, 10000))
#     print(get_cg_iter_time_scipy(A, gt))

def get_pyamgcg_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    repeat=1,
    device="cpu",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    iter_cnt = 0
    time_prec = 0
    time_elp = 0
    b = b.astype(dtype).copy()
    A = A.astype(dtype)

    from pyamg.aggregation import smoothed_aggregation_solver
    from scipy.sparse.linalg import cg

    # Create a PyAMG solver optimized for SPD matrices
    time_start = time.time()
    ml = smoothed_aggregation_solver(A)

    # Create a preconditioner
    M = ml.aspreconditioner(cycle="V")
    time_end = time.time()
    time_prec += time_end - time_start
    # Solve the system multiple times to get accurate timing
    x: np.ndarray = np.zeros_like(b, dtype=dtype)
    for _ in range(repeat):
        x_copy = x.copy()
        b_copy = b.copy()

        iter_cnt = 0

        def counter_callback(x):
            nonlocal iter_cnt
            iter_cnt += 1

        # Time the solve phase
        time_start = time.time()
        _, info = cg(
            A,
            b_copy,
            x0=x_copy,
            rtol=rtol,
            maxiter=max_iter,
            M=M,
            callback=counter_callback,
        )
        time_end = time.time()
        if info != 0:
            iter_cnt = info
        time_elp += time_end - time_start

    return iter_cnt / repeat, time_prec, time_elp / repeat


def get_pyamg_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    repeat=1,
    device="cpu",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    iter_cnt = 0
    time_prec = 0
    time_elp = 0
    b = b.astype(dtype).copy()
    A = A.astype(dtype).sorted_indices()

    from pyamg import smoothed_aggregation_solver as solver

    # Create a PyAMG solver optimized for SPD matrices
    time_start = time.time()
    ml = solver(A)
    time_end = time.time()
    time_prec += time_end - time_start
    # Solve the system multiple times to get accurate timing
    x: np.ndarray = np.zeros_like(b, dtype=dtype)
    for _ in range(repeat):
        x_copy = x.copy()
        b_copy = b.copy()

        iter_cnt = 0

        def counter_callback(x):
            nonlocal iter_cnt
            iter_cnt += 1

        # Time the solve phase
        time_start = time.time()
        ml.solve(b_copy, tol=rtol, maxiter=max_iter, callback=counter_callback)
        time_end = time.time()
        time_elp += time_end - time_start

    return iter_cnt / repeat, time_prec, time_elp / repeat

def get_amgxcg_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    repeat=1,
    device="cpu",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    iter_cnt = 0.0
    time_prec = 0.0
    time_elp = 0.0
    b = b.astype(dtype).copy()
    A = A.astype(dtype)
    import pyamgx

    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict(
        {
            "config_version": 2,
            "solver": {
                "preconditioner": {
                    "print_grid_stats": 0,
                    "print_vis_data": 0,
                    "solver": "AMG",
                    "smoother": {
                        "scope": "jacobi",
                        "solver": "BLOCK_JACOBI",
                        "monitor_residual": 0,
                        "print_solve_stats": 0,
                        "relaxation_factor": 0.6,
                    },
                    "print_solve_stats": 0,
                    "presweeps": 2,
                    "postsweeps": 2,
                    "interpolator": "D2",
                    "max_iters": 1,
                    "monitor_residual": 0,
                    "store_res_history": 0,
                    "scope": "amg",
                    "max_levels": 15,
                    "cycle": "V",
                },
                "solver": "PCG",
                "print_solve_stats": 0,
                "obtain_timings": 0,
                "max_iters": max_iter,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI",
                "scope": "main",
                "tolerance": rtol,
                "norm": "L2",
            },
        }
    )
    
    

    rsc = pyamgx.Resources().create_simple(cfg)

    # Create matrices and vectors:
    A_amgx = pyamgx.Matrix().create(rsc)
    b_amgx = pyamgx.Vector().create(rsc)
    x_amgx = pyamgx.Vector().create(rsc)


    # Create solver:
    solver = pyamgx.Solver().create(rsc, cfg)
    rhs = b.copy()
    sol = np.zeros_like(b, dtype=dtype)
    # Upload system:
    A_amgx.upload_CSR(A)


    # Setup and solve system:
    time_beg = time.time()
    solver.setup(A_amgx)
    time_end = time.time()
    time_prec += time_end - time_beg
    
    
    for _ in range(repeat):
        b_amgx.upload(rhs)
        x_amgx.upload(sol)
        time_beg = time.time()
        solver.solve(b_amgx, x_amgx)
        time_end = time.time()
        time_elp += time_end - time_beg
        iter_cnt += solver.iterations_number

    # Clean up:
    A_amgx.destroy()
    x_amgx.destroy()
    b_amgx.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    return iter_cnt / repeat, time_prec, time_elp / repeat


def get_amgx_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    repeat=1,
    device="cpu",
) -> Tuple[float, float, float]:
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b: np.ndarray = (A @ gt).copy()
    iter_cnt = 0.0
    time_prec = 0.0
    time_elp = 0.0
    b = b.astype(dtype).copy()
    A = A.astype(dtype)
    import pyamgx

    # Initialize config and resources:
    cfg = pyamgx.Config().create_from_dict(
        {
            "config_version": 2,
            "solver": {
                # "print_grid_stats": 1,
                # "print_solve_stats": 1,
                "solver": "AMG",
                "smoother": {
                    "scope": "jacobi",
                    "solver": "BLOCK_JACOBI",
                    "monitor_residual": 0,
                    "print_solve_stats": 0,
                    "relaxation_factor": 0.6,
                },
                "presweeps": 2,
                "interpolator": "D2",
                "obtain_timings": 0,
                "max_iters": max_iter,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI",
                "scope": "main",
                "max_levels": 15,
                "cycle": "W",
                "tolerance": rtol,
                "norm": "L2",
                "postsweeps": 2,
            },
        }
    )

    rsc = pyamgx.Resources().create_simple(cfg)

    # Create matrices and vectors:
    A_amgx = pyamgx.Matrix().create(rsc)
    b_amgx = pyamgx.Vector().create(rsc)
    x_amgx = pyamgx.Vector().create(rsc)


    # Create solver:
    solver = pyamgx.Solver().create(rsc, cfg)
    rhs = b.copy()
    sol = np.zeros_like(b, dtype=dtype)
    # Upload system:
    A_amgx.upload_CSR(A)


    # Setup and solve system:
    time_beg = time.time()
    solver.setup(A_amgx)
    time_end = time.time()
    time_prec += time_end - time_beg
    
    
    for _ in range(repeat):
        b_amgx.upload(rhs)
        x_amgx.upload(sol)
        time_beg = time.time()
        solver.solve(b_amgx, x_amgx)
        time_end = time.time()
        time_elp += time_end - time_beg
        iter_cnt += solver.iterations_number

    # Clean up:
    A_amgx.destroy()
    x_amgx.destroy()
    b_amgx.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()

    return iter_cnt / repeat, time_prec, time_elp / repeat

def get_cupy_iter_time(
    A: csr_matrix,
    gt: np.ndarray,
    rtol=1e-6,
    max_iter=0,
    dtype=np.float64,
    repeat=1,
) -> Tuple[float, float, float]:
    iter_cnt = 0
    time_prec = 0

    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import cg
    # Convert csr_matrix to cupy
    A_cupy = csp.csr_matrix(A)
    gt_cupy = cp.asarray(gt)
    b: np.ndarray = (A_cupy @ gt_cupy).copy()
    rows = A.shape[0]  # type: ignore
    max_iter = max_iter if max_iter > 0 else rows
    b = b.astype(dtype).copy()
    A_cupy = A_cupy.astype(dtype)

    x: np.ndarray = cp.zeros_like(b, dtype=dtype)
    for _ in range(repeat):
        x_copy = x.copy()
        b_copy = b.copy()

        iter_cnt = 0

        def counter_callback(x):
            nonlocal iter_cnt
            iter_cnt += 1

        time_start = time.time()
        _, info = cg(
            A_cupy,
            b_copy,
            x0=x_copy,
            tol=rtol,
            maxiter=max_iter,
            callback=counter_callback,
        )
        time_end = time.time()
        if info != 0:
            iter_cnt += info
        time_prec += time_end - time_start

    return iter_cnt / repeat, 0, time_prec / repeat