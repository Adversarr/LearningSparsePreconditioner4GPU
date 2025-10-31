from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .nn.basic_layers import GraphSpmv


def rel_l2_loss(pred, gt, sqr_out=True, eps=1e-6):
    """Relative L2 Loss

    Args:
        pred (Tensor): prediction, at least 2D
        gt (Tensor): ground truth, at least 2D
        sqr_out (bool, optional): square the output. Defaults to True.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} != {pred.shape}")
    gt_norms = torch.linalg.norm(gt.flatten())
    err_norms = torch.linalg.norm((pred - gt).flatten())
    if sqr_out:
        out = err_norms**2 / (gt_norms**2 + eps)
    else:
        out = err_norms / (gt_norms + eps)
    return out

def l2_loss(pred, gt, sqr_out=True, eps=1e-6):
    """L2 Loss

    Args:
        pred (Tensor): prediction, at least 2D
        gt (Tensor): ground truth, at least 2D
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} != {pred.shape}")
    out = F.mse_loss(pred, gt)
    return out


def nif_loss(pred, gt, sqr_out=True, eps=1e-6):
    """Nif Loss

    Args:
        pred (Tensor): prediction, at least 2D
        gt (Tensor): ground truth, at least 2D
        sqr_out (bool, optional): square the output. Defaults to True.
    """

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} != {pred.shape}")
    gt_norms = torch.linalg.norm(gt.flatten())
    err_norms = torch.linalg.norm((pred - gt).flatten())
    if sqr_out:
        out = err_norms**2 / (gt_norms**2 + eps)
    else:
        out = err_norms / (gt_norms + eps)
    return out


def cosine_similarity_loss(pred, gt, eps=1e-6):
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} != {pred.shape}")
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    csim = F.cosine_similarity(gt_flat, pred_flat, dim=-1, eps=eps).mean()
    return 1 - csim


def cg_alpha(r: torch.Tensor, d: torch.Tensor, q: torch.Tensor):
    """Compute alpha derived from Preconditioned Conjugate Gradient method:
                 dot(r, d)
        alpha = ----------
                 dot(d, q)
    Args:
        r: residual vector
        d: M^-1 r
        q: A d
    """
    assert r.shape == d.shape == q.shape
    r_dot_d = torch.dot(r.flatten(), d.flatten())
    d_dot_q = torch.dot(d.flatten(), q.flatten())
    alpha = r_dot_d / (d_dot_q + 1e-6)
    return alpha


class NeuralCGLossBase(nn.Module):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
    ):
        super().__init__()
        self.spmv = GraphSpmv()
        self.spmv_transpose = GraphSpmv(True)
        self.block_size = block_size
        self.batch_less = batch_less

    # A x = b, M^-1 b = y -> approximated
    def forward(
        self,
        batch,
        d: torch.Tensor,
        L_values: torch.Tensor,
    ):
        """Return the loss value.

        Args:
            batch: Batch of GNN
            d: the preconditioner's output
            L_values: Values of (M^-1)'s factor

        Raises:
            NotImplementedError: If not implemented by the derived class.
        """
        raise NotImplementedError()


class NifLoss(NeuralCGLossBase):
    def __init__(
        self, batch_less: bool, block_size: int, sqr_out: bool = True, eps: float = 1e-6
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(nif_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        n_v = batch.batch.numel()
        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        Ad = self.spmv(
            X=batch.residual, edge_index=batch.edge_index, A=batch.matrix_values
        )
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            d_sample = d[b:e]
            gt_sample = Ad[b:e]
            loss = loss + self.fn(d_sample, gt_sample)
        bsize = len(ptrs) - 1
        return loss / bsize


class RelativeL2Loss_PlainNorm(NeuralCGLossBase):
    def __init__(
        self, batch_less: bool, block_size: int, sqr_out: bool = True, eps: float = 1e-6
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        # return self.fn(pred=self.to_batched(batch.gt, batch), gt=self.to_batched(x))
        gt = batch.gt
        n_v = batch.batch.numel()
        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            d_sample = d[b:e]
            gt_sample = gt[b:e]
            loss = loss + self.fn(d_sample, gt_sample)
        bsize = len(ptrs) - 1
        return loss / bsize


class RelativeL2Loss_ANorm(NeuralCGLossBase):
    def __init__(
        self, batch_less: bool, block_size: int, sqr_out: bool = True, eps: float = 1e-6
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        n_v = batch.batch.numel()
        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            d_sample = Ad[b:e]
            gt_sample = batch.residual[b:e]
            loss = loss + self.fn(d_sample, gt_sample)
        bsize = len(ptrs) - 1
        return loss / bsize

class L2Loss_ANorm(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        return F.mse_loss(Ad, batch.residual)

class CosineSimilarityLoss_PlainNorm(NeuralCGLossBase):
    def __init__(self, batch_less: bool, block_size: int, eps: float = 1e-6):
        super().__init__(batch_less, block_size)
        self.fn = partial(cosine_similarity_loss, eps=eps)

    def forward(self, batch, d, L_values):
        gt = batch.gt
        n_v = batch.batch.numel()
        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            d_sample = d[b:e]
            gt_sample = gt[b:e]
            loss = loss + self.fn(d_sample, gt_sample)
        bsize = len(ptrs) - 1
        return loss / bsize


class CosineSimilarityLoss_ANorm(NeuralCGLossBase):
    def __init__(self, batch_less: bool, block_size: int, eps: float = 1e-6):
        super().__init__(batch_less, block_size)
        self.fn = partial(cosine_similarity_loss, eps=eps)

    def forward(self, batch, d, L_values):
        n_v = batch.batch.numel()
        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            d_sample = Ad[b:e]
            gt_sample = batch.residual[b:e]
            loss = loss + self.fn(d_sample, gt_sample)
        bsize = len(ptrs) - 1
        return loss / bsize


class ConjGradLoss_PlainNorm(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        r = batch.residual
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        gt = batch.gt
        n_v = batch.batch.numel()

        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            r_sample = r[b:e]
            d_sample = d[b:e]
            Ad_sample = Ad[b:e]
            alpha = cg_alpha(r_sample, d_sample, Ad_sample)
            gt_sample = gt[b:e]
            loss = loss + self.fn(alpha * d_sample, gt_sample)
        bsize = len(ptrs) - 1
        return loss / bsize


class ConjGradLoss_ANorm(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        r = batch.residual
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        n_v = batch.batch.numel()

        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            r_sample = r[b:e]
            d_sample = d[b:e]
            Ad_sample = Ad[b:e]
            alpha = cg_alpha(r_sample, d_sample, Ad_sample)
            loss = loss + self.fn(alpha * Ad_sample, r_sample)
        bsize = len(ptrs) - 1
        return loss / bsize

class ConjGradLoss_ANorm_NoRelative(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        r = batch.residual
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        n_v = batch.batch.numel()

        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            r_sample = r[b:e]
            d_sample = d[b:e]
            Ad_sample = Ad[b:e]
            alpha = cg_alpha(r_sample, d_sample, Ad_sample)
            loss = loss + self.fn(alpha * Ad_sample, r_sample)
        bsize = len(ptrs) - 1
        return loss / bsize


class PropLoss(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        n_v = batch.batch.numel()

        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            r_sample = batch.residual[b:e]
            Ad_sample = Ad[b:e]
            nrm2_Ad = torch.dot(Ad_sample.flatten(), Ad_sample.flatten())
            Ad_r = torch.dot(Ad_sample.flatten(), r_sample.flatten())
            nrm2_r = torch.dot(r_sample.flatten(), r_sample.flatten())

            loss_sample = (nrm2_Ad - Ad_r * Ad_r / nrm2_r)
            loss = loss + loss_sample
        return F.mse_loss(Ad, batch.residual)


class RelPropLoss(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = partial(rel_l2_loss, sqr_out=sqr_out, eps=eps)

    def forward(self, batch, d, L_values):
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        n_v = batch.batch.numel()

        if self.batch_less:
            ptrs: list[int] = [0, n_v]
        else:
            ptrs = batch.ptr.detach().cpu().numpy().tolist()
        loss = 0.0
        for b, e in zip(ptrs[:-1], ptrs[1:]):
            r_sample = batch.residual[b:e]
            Ad_sample = Ad[b:e]
            nrm2_Ad = torch.dot(Ad_sample.flatten(), Ad_sample.flatten())
            Ad_r = torch.dot(Ad_sample.flatten(), r_sample.flatten())
            nrm2_r = torch.dot(r_sample.flatten(), r_sample.flatten())

            loss_sample = (nrm2_Ad - Ad_r * Ad_r / nrm2_r) / nrm2_r
            loss = loss + loss_sample
        return F.mse_loss(Ad, batch.residual)

class L1Loss(NeuralCGLossBase):
    def __init__(
        self,
        batch_less: bool,
        block_size: int,
        sqr_out: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(batch_less, block_size)
        self.fn = nn.L1Loss(reduction="mean")

    def forward(self, batch, d, L_values):
        Ad = self.spmv(X=d, edge_index=batch.edge_index, A=batch.matrix_values, mask=batch.mask)
        return self.fn(Ad, batch.residual)


def create_loss_item(name: str, batch_less, block_size, **params) -> NeuralCGLossBase:
    name_lower = name.lower()
    if name_lower == "relativel2loss_plainnorm":
        return RelativeL2Loss_PlainNorm(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "relativel2loss_anorm":
        return RelativeL2Loss_ANorm(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "l2loss_anorm":
        return L2Loss_ANorm(batch_less=batch_less, block_size=block_size, **params)
    elif name_lower == "proploss":
        return PropLoss(batch_less=batch_less, block_size=block_size, **params)
    elif name_lower == "l1loss":
        return L1Loss(batch_less=batch_less, block_size=block_size, **params)
    elif name_lower == "relproploss":
        return RelPropLoss(batch_less=batch_less, block_size=block_size, **params)
    elif name_lower == "cosinesimilarityloss_plainnorm":
        return CosineSimilarityLoss_PlainNorm(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "cosinesimilarityloss_anorm":
        return CosineSimilarityLoss_ANorm(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "conjgradloss_plainnorm":
        return ConjGradLoss_PlainNorm(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "conjgradloss_anorm":
        return ConjGradLoss_ANorm(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "conjgradloss_anorm_norelative":
        return ConjGradLoss_ANorm_NoRelative(
            batch_less=batch_less, block_size=block_size, **params
        )
    elif name_lower == "nifloss_norm":
        return NifLoss(batch_less=batch_less, block_size=block_size, **params)
    else:
        raise ValueError(f"Unknown loss {name}")
