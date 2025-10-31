from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from pymathprim.linalg.cg_host import ainv, ichol
from ilupp import ichol0
from tqdm import tqdm

from neural_cg.utils.datamodule import FolderDataModule
from neural_cg.utils.validate import to_csr_cpu

def geometric_mean(x: np.ndarray):
    """
    Compute the geometric mean of a vector x.
    """
    return np.exp(np.log(x).mean())

def condition_number(
    A: np.ndarray,
    M: Union[np.ndarray, None] = None,
) -> Tuple[float, float]:
    """
    Compute the condition number of a matrix A with respect to a preconditioner M.
    """
    assert A.shape[0] == A.shape[1]  # == M.shape[0] == M.shape[1]
    MA = M @ A if M is not None else A
    eval = np.abs(np.linalg.eigvalsh(MA))
    minimum = eval.min()
    maximum = eval.max()
    standard = maximum / minimum
    trace = np.mean(eval)
    geom = geometric_mean(eval)
    kaporin = trace / geom
    return standard, kaporin

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
    workspace_name = cfg.get("workspace", "simple")
    logger.info(f"Using workspace: {workspace_name}")
    workspace_class = get_workspace(workspace_name)
    # Load the model.
    model = workspace_class(
        edge_features=edge_input_features,
        node_features=node_input_features,
        block_size=block_size,
        **cfg,  # type: ignore
    )
    model = workspace_class.load_from_checkpoint(cfg.pretrained)
    logger.info(f"Loaded pretrained model from {cfg.pretrained}")
    device = "cpu"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device = "cuda"
    model = model.to(device)
    model.eval()
    
    stat_dict: Dict[str, List[float]] = {
        'neural': [],
        'none': [],
        'diag': [],
        'ainv': [],
        'ichol': []
    }
    
    stat_dict_kaporin: Dict[str, List[float]] = {
        'neural': [],
        'none': [],
        'diag': [],
        'ainv': [],
        'ichol': []
    }

    logger.info(f"epsilon: {model.epsilon}")
    visualize = cfg.get("visualize", True)
    with torch.no_grad():
        # Warmup the neural network:
        sample = data.dataset.get(0)
        sample = sample.to(device)
        try:
            for sample in tqdm(data.test_dataloader(), desc="Inference"):
                sample = sample.to(device)
                mat_size = (sample.ptr[-1] * model.block_size).item()
                mask = sample.mask
                A = to_csr_cpu(sample.edge_index, sample.matrix_values, mat_size, mask, dtype=np.float64)
                Adense = A.todense()

                # Preconditioner.
                # L, _ = model.inference_step(sample)
                # M = L @ L.T
                # Mdense = M.todense() + model.epsilon * np.eye(M.shape[0], dtype=np.float64)
                Mdense = model.precondition(sample, Adense)

                # Compute condition number of neural.
                cond_num = condition_number(Adense, Mdense)
                del Mdense
                # Compute condition number of CG.
                cond_num_original = condition_number(Adense)

                # Compute condition number of diag.
                diags = np.diag(Adense)
                cond_num_diag = condition_number(Adense, np.diag(1.0 / diags))
                del diags

                # Compute condition number of ainv.
                ai = ainv(A)
                aidense = (ai @ ai.T).todense()  # A^{-1} = L @ L.T
                cond_num_ai = condition_number(Adense, aidense)
                del ai

                # Compute condition number of ichol.
                ic = ichol(A).todense()  # A = L @ L.T => M = (L @ L.T)^{-1}
                M = np.linalg.inv(ic @ ic.T)
                cond_num_ic = condition_number(Adense, M)
                del ic

                # Store the condition numbers.
                stat_dict['neural'].append(cond_num[0])
                stat_dict['none'].append(cond_num_original[0])
                stat_dict['diag'].append(cond_num_diag[0])
                stat_dict['ainv'].append(cond_num_ai[0])
                stat_dict['ichol'].append(cond_num_ic[0])
                # Store the kaporin condition numbers.
                stat_dict_kaporin['neural'].append(cond_num[1])
                stat_dict_kaporin['none'].append(cond_num_original[1])
                stat_dict_kaporin['diag'].append(cond_num_diag[1])
                stat_dict_kaporin['ainv'].append(cond_num_ai[1])
                stat_dict_kaporin["ichol"].append(cond_num_ic[1])

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, stop testing.")
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            raise e

        exp_name = cfg.exp_name
        def do_save(stat_dict, name):
            table = pd.DataFrame(stat_dict)
            table.to_csv(f"{name}_cond_{exp_name}.csv", index=False)
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=table)
            plt.title("Condition Number Distribution")
            plt.ylabel("Condition Number")
            if name == 'cond':
                plt.yscale('log')
            plt.savefig(f"{name}_cond_{exp_name}.png", dpi=300)
            if visualize:
                plt.show()
        do_save(stat_dict, "cond")
        do_save(stat_dict_kaporin, "kaporin")


if __name__ == "__main__":
    main()
