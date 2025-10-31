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
from preprocess.heatmultisource import train_dataset, test_dataset, HeatDatasetMultiSource

class NeuralPCG_Dataset(Dataset):
    def __init__(self, pcgset: HeatDatasetMultiSource, use_random_rhs=False):
        super().__init__()
        self.pcgset = pcgset
        self.use_random_rhs = use_random_rhs

    def len(self):
        return len(self.pcgset)

    def get(self, idx):
        data = self.pcgset[idx]
        rhs = data.rhs
        diag = data.diag
        gt = data.u_next
        mask = 1 - data.x[:, [3]]  # one-hot mask for DBCs, dbc=0, non-dbc=1

        inv_diag = 1 / diag
        rsqrt_diag = torch.sqrt(1 / diag)

        if self.use_random_rhs:
            rhs = torch.randn_like(rhs) * mask

        matrix_values = (data.edge_attr[:, -1] + data.edge_attr[:, -2]) * 0.5

        OOD = True
        if OOD:
            diag_idx = data.edge_index[0, :] == data.edge_index[1, :]
            matrix_values[diag_idx] -= 1e-1
            print(f"Number of diagonal entries: {diag_idx.sum()}")

        return Data(
            x=data.x,
            mask=mask,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            matrix_values=matrix_values.reshape(-1, 1, 1),
            residual=rhs,
            gt=gt,
            diagonal=diag,
            inv_diag=inv_diag,
            rsqrt_diag=rsqrt_diag,
        )

class NeuralPCG_DataModule(L.LightningDataModule):
    def __init__(self, data_config, split_config, batch_size):
        super().__init__()
        self.trainset = NeuralPCG_Dataset(train_dataset(), use_random_rhs=data_config.use_random_rhs)
        self.valset = NeuralPCG_Dataset(test_dataset(), use_random_rhs=data_config.use_random_rhs)
        logger.warning("Ignoring split_config.")
        self.batch_size = batch_size
        self.dataset = self.trainset.pcgset
        logger.info(f"Train set size: {len(self.trainset)}")
        logger.info(f"Validation set size: {len(self.valset)}")
    @override
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    @override
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)

    @override
    def test_dataloader(self):
        return self.val_dataloader()

def get_workspace(name):
    if name == "simple":
        from neural_cg.workspace import SimpleTrainingWorkspace
        return SimpleTrainingWorkspace
    elif name == 'scaled':
        from neural_cg.scaled_workspace import ScaledTrainingWorkspace
        return ScaledTrainingWorkspace
    else:
        raise ValueError(f"Unknown workspace name: {name}")

@hydra.main(config_path="config", config_name="basic", version_base="1.3")
def main(cfg: DictConfig):
    cfg.exp_name = 'heatmultisource'
    L.seed_everything(cfg.seed, workers=True)
    data = NeuralPCG_DataModule(
        data_config=cfg.data,
        split_config=cfg.split,
        batch_size=cfg.batch_size,
    )

    edge_input_features = 3
    node_input_features = 4
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

    trainer = L.Trainer(
        logger=MLFlowLogger(cfg.exp_name),
        callbacks=[
            LearningRateMonitor(),
            ModelSummary(3),

            ModelCheckpoint(save_last=True, every_n_epochs=10),
        ],
        **cfg.trainer,
    )
    if "no_train" in cfg and cfg.no_train:
        logger.info("Skipping training!!!")
    else:
        trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
