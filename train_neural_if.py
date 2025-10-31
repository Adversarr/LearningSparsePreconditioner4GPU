from pathlib import Path
from typing import override
import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data, DataLoader, Dataset

from neural_cg.utils.weight_init import weight_init


class NIF_Dataset(Dataset):
    def __init__(self, folder: Path, use_random_rhs: bool = False):
        super().__init__()
        self.folder = Path(folder)
        assert self.folder.exists()
        files = list(self.folder.glob("*.pt"))
        assert len(files) > 0
        self.all_data = [torch.load(f, weights_only=False) for f in files]
        self.use_random_rhs = use_random_rhs
        assert use_random_rhs
        logger.info(f"Loaded {len(self.all_data)} files from {self.folder}")

    def len(self):
        return len(self.all_data)

    def get(self, idx):
        data = self.all_data[idx]
        mask = torch.ones_like(data.x)  # [10000, 1]
        rhs = torch.randn_like(data.x)  # [10000, 1]
        matrix_values = data.edge_attr  # [nE, 1]

        def extract_diagonal(i, j, val):
            mask = i == j

            diagonal_indices = i[mask]
            diagonal_values = val[mask]

            sorted_indices = torch.argsort(diagonal_indices)
            return diagonal_indices[sorted_indices], diagonal_values[sorted_indices]

        diagonal_indices, diagonal_values = extract_diagonal(
            data.edge_index[0], data.edge_index[1], data.edge_attr
        )

        diag = diagonal_values
        inv_diag = torch.reciprocal(diag)
        rsqrt_diag = torch.rsqrt(diag)
        return Data(
            x=data.x,
            mask=mask,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            matrix_values=matrix_values.reshape(-1, 1, 1),
            residual=rhs,
            diagonal=diag,
            inv_diag=inv_diag,
            rsqrt_diag=rsqrt_diag,
        )


class NIF_DataModule(L.LightningDataModule):
    def __init__(self, data_path: Path, data_config, split_config, batch_size):
        super().__init__()
        self.trainset = NIF_Dataset(
            data_path / "train", use_random_rhs=data_config.use_random_rhs
        )
        self.valset = NIF_Dataset(
            data_path / "val", use_random_rhs=data_config.use_random_rhs
        )
        self.testset = NIF_Dataset(
            data_path / "test", use_random_rhs=data_config.use_random_rhs
        )
        self.dataset = self.trainset
        logger.warning("Ignoring split_config.")
        self.batch_size = batch_size
        logger.info(f"Train set size: {len(self.trainset)}")
        logger.info(f"Validation set size: {len(self.valset)}")

    @override
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    @override
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, shuffle=False)

    @override
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=1, shuffle=False)


def get_workspace(name):
    if name == "simple":
        from neural_cg.workspace import SimpleTrainingWorkspace

        return SimpleTrainingWorkspace
    elif name == "scaled":
        from neural_cg.scaled_workspace import ScaledTrainingWorkspace

        return ScaledTrainingWorkspace
    else:
        raise ValueError(f"Unknown workspace name: {name}")


@hydra.main(config_path="config", config_name="basic", version_base="1.3")
def main(cfg: DictConfig):
    try:
        import rich

        has_rich = True
    except ImportError:
        has_rich = False
        logger.warning(
            "Rich library is not installed. Some features may be unavailable."
        )

    cfg.exp_name = "neuralif"
    L.seed_everything(cfg.seed, workers=True)
    data = NIF_DataModule(
        data_path=Path(cfg.get("data_path", "data/Random")),
        data_config=cfg.data,
        split_config=cfg.split,
        batch_size=cfg.batch_size,
    )

    edge_input_features = 1
    node_input_features = 1
    block_size = 1
    logger.info(f"Edge input features: {edge_input_features}")
    logger.info(f"Node input features: {node_input_features}")
    logger.info(f"Block size: {block_size}")

    workspace_name = cfg.get("workspace", "simple")
    logger.info(f"Using workspace: {workspace_name}")
    workspace_class = get_workspace(workspace_name)
    model = workspace_class(
        edge_features=edge_input_features,
        node_features=node_input_features,
        block_size=block_size,
        **cfg,  # type: ignore
    )

    pretrained_path = ""
    if "pretrained" in cfg:
        pretrained_path = str(cfg.pretrained)
    if len(pretrained_path) != 0:
        model = workspace_class.load_from_checkpoint(pretrained_path)
        logger.info(f"Loaded pretrained model from {pretrained_path}")
    else:
        model.apply(weight_init)
        logger.info("Initialized model.")

    if torch.cuda.is_available():
        tensor_core_precision = cfg.get("tensor_core_precision", "high")
        torch.set_float32_matmul_precision(tensor_core_precision)
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(save_last=True, every_n_epochs=10),
        # ModelSummary(3),
    ]
    if has_rich:
        from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar

        callbacks.append(RichModelSummary(max_depth=3))
        callbacks.append(RichProgressBar())
    else:
        callbacks.append(ModelSummary(max_depth=3))

    trainer = L.Trainer(
        logger=MLFlowLogger(cfg.exp_name),
        callbacks=callbacks,
        **cfg.trainer,
    )
    if "no_train" in cfg and cfg.no_train:
        logger.info("Skipping training!!!")
    else:
        trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
