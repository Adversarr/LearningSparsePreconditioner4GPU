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

from neural_cg.utils.datamodule import FolderDataModule, MultiFolderDataModule
from neural_cg.utils.weight_init import weight_init


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
    from importlib.util import find_spec
    has_rich = find_spec("rich") is not None

    L.seed_everything(cfg.seed, workers=True)

    # Decide which datamodule to use (single or multi-folder)
    use_multidata = 'all_prefix' in cfg.data
    logger.info(f"Use multidata: {use_multidata}")

    DataModule = MultiFolderDataModule if use_multidata else FolderDataModule
    data = DataModule(
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

    workspace_name = cfg.get("workspace", "simple")
    logger.info(f"Using workspace: {workspace_name}")
    workspace_class = get_workspace(workspace_name)
    model = workspace_class(
        edge_features=edge_input_features,
        node_features=node_input_features,
        block_size=block_size,
        **cfg,  # type: ignore
    )

    pretrained_path: str = ""
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

    # Configure checkpoint filename (optional per-mode)
    exp_name = cfg.exp_name
    prefix = f"{exp_name}-{workspace_name}-"
    checkpoint_filename = prefix + "{epoch:02d}"

    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=cfg.checkpoint.get("every_n_epochs", 1),
            filename=checkpoint_filename,
        ),
    ]
    summary_depth = 1
    if has_rich:
        from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
        callbacks.append(RichModelSummary(max_depth=summary_depth))
        callbacks.append(RichProgressBar())
    else:
        callbacks.append(ModelSummary(max_depth=summary_depth))

    trainer = L.Trainer(
        logger=MLFlowLogger(exp_name),
        callbacks=callbacks,
        **cfg.trainer,
    )

    logger.info(f"Dataset Length: {len(data.dataset)}")
    if cfg.get("no_train", False):
        logger.info("Skipping training!!!")
    else:
        trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()