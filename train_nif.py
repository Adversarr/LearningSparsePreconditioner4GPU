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

from neural_cg.utils.datamodule import FolderDataModule
from neural_cg.utils.weight_init import weight_init
from neural_cg.nif import NeuralPCG, NeuralIncompleteFactorization
def get_workspace(name):
    if name == "npcg":
        return NeuralPCG
    elif name == 'nif':
        return NeuralIncompleteFactorization
    else:
        raise ValueError(f"Unknown workspace name: {name}")

@hydra.main(config_path="config", config_name="basic", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    if str(cfg.loss.name).lower() != "nifloss_norm":
        logger.warning("Loss function in config is not NIFLossNorm. Enforcing...")
        cfg.loss.name = "NifLoss_Norm"

    data = FolderDataModule(
        data_config=cfg.data,
        split_config=cfg.split,
        batch_size=cfg.batch_size,
    )

    edge_input_features = data.dataset.num_edge_features
    node_input_features = data.dataset.num_node_features
    block_size = data.dataset.block_size
    logger.info(f"Edge input features: {edge_input_features}")
    logger.info(f"Node input features: {node_input_features}")
    logger.info(f"Block size: {block_size}")

    workspace_name = cfg.get("workspace", "npcg")
    logger.info(f"Using workspace: {workspace_name}")
    workspace_class = get_workspace(workspace_name)
    
    
    model = workspace_class(
        edge_features=edge_input_features,
        node_features=node_input_features,
        block_size=block_size,
        drop_tol=0.0,
        **cfg,  # type: ignore
    )

    pretrained_path = ""
    if "pretrained" in cfg:
        pretrained_path = str(cfg.pretrained)
    if len(pretrained_path) != 0:
        model = workspace_class.load_from_checkpoint(pretrained_path)
        # model = NeuralIncompleteFactorization.load_from_checkpoint(pretrained_path)
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
