from time import time
from typing import List, Union

import lightning as L
import numpy as np
import torch
from loguru import logger
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

from neural_cg.utils.optim import create_optimizer, create_scheduler
from neural_cg.utils.validate import (
    get_cg_iter_time,
    get_pcg_iter_time,
    get_pcg_iter_time_scipy,
    get_pcg_stat_scipy,
    to_csr_cpu,
    to_numpy,
)

from .loss import create_loss_item
from .nn.basic_layers import AATPE
from .nn.gnns import NodeEdgeProcessing


class SimpleTrainingWorkspace(L.LightningModule):
    def __init__(
        self,
        # Trainer
        batch_size: int,
        batch_less: bool,
        block_size: int,
        test_max_iter: int,
        optimizer,
        scheduler,
        loss,
        inspect_norms: bool,
        # Convergence
        check_converge: bool,
        check_methods: List[str],
        check_devices: List[str],
        # Neural network related.
        node_features: int,
        edge_features: int,
        gnn,
        epsilon: float,
        # Other module's hparams
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.block_size = block_size
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.epsilon = epsilon
        self.test_max_iter = test_max_iter

        #### build loss
        self.loss_name = loss.name
        self.loss_fn = create_loss_item(
            loss.name,
            batch_less=batch_less,
            block_size=block_size,
            **(loss.params or {}),
        )

        #### build neural network
        self.gnn = NodeEdgeProcessing(
            node_in_features=node_features,
            node_out_features=None,
            edge_in_features=edge_features,
            edge_out_features=block_size * block_size,
            **gnn,
        )  # type: ignore

        self.enable_inspect_norms = inspect_norms
        self.spai_preconditioner = AATPE(self.epsilon)
        self.check_converge = check_converge
        self.check_methods: List[str] = check_methods
        self.check_devices: List[str] = check_devices

        try:
            import pymathprim
            self.has_pymathprim = True
        except ImportError:
            self.has_pymathprim = False
        if check_converge and not self.has_pymathprim:
            logger.warning("pymathprim is not installed, convergence check will fallback to scipy.")

    def forward(self, node_attr, edge_index, edge_attr):
        _, boo_entry = self.gnn(node_attr, edge_index, edge_attr)
        return boo_entry.reshape(-1, self.block_size, self.block_size)  # [nE, b, b]

    def _shared_step(self, batch: Data, prefix: str, batch_idx: int) -> torch.Tensor:
        assert hasattr(batch, "ptr"), "Relying on batch.ptr to compute batch size"
        bsize = batch.ptr.shape[0] - 1
        boo_entry = self(batch.x, batch.edge_index, batch.edge_attr)
        d = self.spai_preconditioner(batch.residual, batch.edge_index, boo_entry, mask=batch.mask)

        loss_dict = {}
        total_loss = self.loss_fn(batch, d, boo_entry)
        loss_dict[f"{prefix}/Loss"] = total_loss
        # loss_dict[f"{prefix}/{self.loss_name}"] = total_loss
        self.log_dict(loss_dict, batch_size=bsize, prog_bar=(prefix == "Train"))

        if batch_idx == 0 and prefix in ["Val", "Test"]:
            self.log_converge_batch(boo_entry, batch, prefix)

        self._log_norms(boo_entry, bsize, prefix)
        return boo_entry, total_loss  # type: ignore

    def log_converge_batch(self, boo_entry: torch.Tensor, batch: Data, prefix: str, repeat=1):
        if not self.check_converge:
            return

        ptr = batch.ptr.detach().cpu().numpy().tolist()
        mask = batch.mask
        n = ptr[-1] * self.block_size
        L: csr_matrix = to_csr_cpu(
            batch.edge_index,  # type: ignore
            boo_entry,
            n,
            mask,
            dtype=np.float64,  # type: ignore
        )
        A: csr_matrix = to_csr_cpu(
            batch.edge_index,  # type: ignore
            batch.matrix_values,
            n,
            mask,
            dtype=np.float64,  # type: ignore
        )
        # r = to_numpy(batch.residual).flatten().astype(np.float64, copy=True)
        r = to_numpy(batch.mask).flatten().astype(np.float64, copy=True)
        for b, e in zip(ptr[:-1], ptr[1:]):
            b = b * self.block_size
            e = e * self.block_size
            r_sub = r[b:e]
            A_sub = A[b:e, b:e].copy()
            L_sub = L[b:e, b:e].copy()
            self.log_converge_v2(L_sub, A_sub, r_sub, repeat, prefix)

    def log_converge_v2(self, L, A, r, repeat, prefix: str):
        if not self.has_pymathprim:
            return self.log_converge_scipy(L, A, r, repeat, prefix)

        stats = {}

        def put_stats(name, value):
            if name in stats:
                stats[name] += value
            else:
                stats[name] = value

        for device in self.check_devices:
            for method in self.check_methods:
                avg_iter, avg_prec, avg_solve = get_cg_iter_time(A, r, repeat=repeat, method=method)
                put_stats(f"{prefix}/{device}_{method}_iter", avg_iter)
                put_stats(f"{prefix}/{device}_{method}_solve", avg_solve)
            avg_iter, _, avg_solve = get_pcg_iter_time(A, r, L, self.epsilon, device=device, repeat=repeat)
            # The precompute step is not considered, just ignore.
            put_stats(f"{prefix}/{device}_neural_iter", avg_iter)
            put_stats(f"{prefix}/{device}_neural_solve", avg_solve)
        self.log_dict(stats, batch_size=1)

    def log_converge_scipy(self, L, A, r, repeat, prefix: str):
        stats = get_pcg_stat_scipy(A, r, prefix)
        stats[f"{prefix}/cpu_neural_iter"] = get_pcg_iter_time_scipy(A, r, L, self.epsilon)
        self.log_dict(stats, batch_size=1)

    def _log_norms(self, boo_entry: torch.Tensor, bsize: int, prefix: str):
        if self.enable_inspect_norms:
            norm = torch.linalg.norm(boo_entry)
            self.log(f"{prefix}/boo_entry_norm", norm, batch_size=1)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        _, loss = self._shared_step(batch, "Train", batch_idx)
        return loss  # type: ignore

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        boo_entry, loss = self._shared_step(batch, "Val", batch_idx)
        if batch_idx == 0:
            self.log_converge_batch(boo_entry, batch, "Val", 1)
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        boo_entry, loss = self._shared_step(batch, "Test", batch_idx)
        self.log_converge_batch(boo_entry, batch, "Test", 3)
        _, dt = self.inference_step(batch)
        self.log("Test/time_neural_prec", dt, batch_size=1)
        return loss

    def inference_step(self, sample: Data, time_beg: Union[float, None] = None):
        edge_index = sample.edge_index
        edge_attr = sample.edge_attr
        mask = sample.mask
        x = sample.x
        time_beg = time()
        boo_entry = self(x, edge_index, edge_attr).cpu().detach().numpy() # type: ignore
        dt = time() - time_beg # now, BCOO matrix is computed. This is the preconditioner time.
        n = x.shape[0] * self.block_size  # type: ignore
        csr = to_csr_cpu(edge_index, boo_entry, n, mask, dtype=np.float64)  # type: ignore
        return csr, dt

    def precondition(self, sample: Data, Adense) -> np.ndarray:
        L, _ = self.inference_step(sample)
        M = L @ L.T
        Mdense = M.todense() + self.epsilon * np.eye(M.shape[0], dtype=np.float64)
        return Mdense

    def configure_optimizers(self):  # type: ignore
        opt = create_optimizer(
            self.parameters(),
            self.optimizer_cfg.name,
            self.optimizer_cfg.params,
        )
        sched = create_scheduler(
            opt,
            self.scheduler_cfg.name,
            self.scheduler_cfg.params,
        )
        if sched is not None:
            return {"optimizer": opt, "lr_scheduler": sched}
        else:
            return opt

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        if self.global_step % 100 > 0:
            return
        total_norm = 0.0
        total_grad_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad.data, 2)
                param_norm = torch.norm(param.data, 2)
                total_grad_norm += grad_norm.item() ** 2
                total_norm += param_norm.item() ** 2
        self.log_dict(
            {
                "Train/total_grad_norm": total_grad_norm**0.5,
                "Train/total_param_norm": total_norm**0.5,
            }
        )
