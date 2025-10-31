from dataclasses import dataclass
from typing import Dict, List
import hydra
import ilupp
import torch
from loguru import logger
from omegaconf import DictConfig
import numpy as np
from neural_cg.utils.datamodule import FolderDataModule
from neural_cg.utils.validate import (
    get_cg_iter_time,
    get_pcg_ic_iter_time_scipy,
    get_pcg_iter_time,
    get_pcg_iter_time_scipy,
    get_pcg_iter_time_scipy_ichol,
    get_pcg_scaled_iter_time,
    to_csr_cpu,
    to_numpy,
)
from tqdm import tqdm, trange

@dataclass
class InferenceTimestat:
    all_solve_time: List[float]
    all_prec_time: List[float]
    all_iteration: List[float]
    all_matrix_size: List[int]

class Timestat:
    def __init__(self):
        self.stat_dict:Dict[str, InferenceTimestat] = {}

    def put(self, key: str, solve_time: float, prec_time: float, iteration: float, all_matrix_size: int):
        if key not in self.stat_dict:
            self.stat_dict[key] = InferenceTimestat([], [], [], [])
        self.stat_dict[key].all_solve_time.append(solve_time)
        self.stat_dict[key].all_prec_time.append(prec_time)
        self.stat_dict[key].all_iteration.append(iteration)
        self.stat_dict[key].all_matrix_size.append(all_matrix_size)

    def rich_print(self):
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Inference Time Statistics")
        table.add_column("Key", justify="left")
        table.add_column("Total Time (ms)", justify="right")
        table.add_column("Solve Time (ms)", justify="right")
        table.add_column("Precond Time (ms)", justify="right")
        table.add_column("#Iteration", justify="right")

        for key, stat in self.stat_dict.items():
            avg_solve_time = np.mean(stat.all_solve_time) * 1000
            avg_prec_time = np.mean(stat.all_prec_time) * 1000
            avg_iteration = np.mean(stat.all_iteration)
            row = [
                key,
                f"{avg_solve_time + avg_prec_time:.2f}",
                f"{avg_solve_time:.2f}",
                f"{avg_prec_time:.2f}",
                f"{avg_iteration:.4f}",
            ]
            table.add_row(*row)

        console.print(table)

    def print(self):
        for key, stat in self.stat_dict.items():
            avg_solve_time = np.mean(stat.all_solve_time) * 1000
            avg_prec_time = np.mean(stat.all_prec_time) * 1000
            avg_iteration = np.mean(stat.all_iteration)
            logger.info(
                f"{key}: Average Solve Time: {avg_solve_time:.2f} ms"
                f"Precond Time: {avg_prec_time:.2f} ms"
                f"Iteration Time: {avg_iteration:.4f} it/sample"
            )

def get_workspace(name):
    if name == "simple":
        from neural_cg.workspace import SimpleTrainingWorkspace
        return SimpleTrainingWorkspace, get_pcg_iter_time
    elif name == 'scaled':
        from neural_cg.scaled_workspace import ScaledTrainingWorkspace
        return ScaledTrainingWorkspace, get_pcg_scaled_iter_time
    else:
        raise ValueError(f"Unknown workspace name: {name}")

@hydra.main(config_path="config", config_name="basic", version_base="1.3")
def main(cfg: DictConfig):
    from train_neural_pcg_heat import NeuralPCG_DataModule
    cfg.exp_name = "heatmultisource"
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
    workspace_class, pcg = get_workspace(workspace_name)

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

    repeat = cfg.get("repeat", 1)
    rhs = cfg.get("rhs", 'mask') # use mask as rhs

    logger.info("Repeat each sample for {} times.".format(repeat))

    rtol = cfg.get("rtol", 1e-6)
    logger.info(f"Relative tolerance: {rtol}")

    with torch.no_grad():
        stats = Timestat()

        # Warmup the neural network:
        sample = data.trainset.get(0)
        sample = sample.to(device)
        for _ in trange(cfg.get("warmup", 20), desc="Warmup Torch"):
            model.inference_step(sample)

        try:
            def do_inference(sample):
                sample = sample.to(device)
                mat_size = (sample.ptr[-1] * model.block_size).item()
                mask = sample.mask
                A = to_csr_cpu(sample.edge_index, sample.matrix_values, mat_size, mask)
                A_full = to_csr_cpu(sample.edge_index, sample.matrix_values, mat_size, None)
                A_full.data.fill(1.0)

                # Preconditioner Time.
                prec = 0.
                for _ in range(repeat):
                    _, this_prec = model.inference_step(sample)
                    prec += this_prec
                prec /= repeat

                # CG Time.
                L, _ = model.inference_step(sample)
                m =  to_numpy(mask).flatten().astype(np.float64)
                r = to_numpy(sample.gt).flatten().astype(np.float64)
                assert not np.any(np.isnan(L.data))
                assert not np.any(np.isnan(A.data))
                assert not np.any(np.isnan(r))
                try:
                    msize = A.shape[0]
                    it, dt = get_pcg_iter_time_scipy(A, r, L, model.epsilon,rtol=rtol, with_time=True)
                    stats.put(f"PCG_Scipy_{msize}", dt, prec, it, msize)
                    it, _, dt = get_pcg_iter_time(A, r, L, model.epsilon, rtol=rtol, device='cuda')
                    stats.put(f"PCG_{msize}_cuda", dt, prec, it, msize)
                    it, _, dt = get_pcg_iter_time(A, r, L, model.epsilon, rtol=rtol, device='cuda')
                    stats.put(f"PCG_{msize}_cpu", dt, prec, it, msize)
                    # it, dt = get_pcg_ic_iter_time_scipy(A, r, rtol=rtol, with_time=True)
                    it, prec, dt = get_cg_iter_time(A, r, rtol=rtol, device='cuda', method='ic')
                    stats.put(f"PCG_IC_mp_{msize}", dt, prec, it, msize)
                    it, prec, dt = get_cg_iter_time(A, r, rtol=rtol, device='cuda', method='diagonal')
                    stats.put(f"PCG_Diag_mp_{msize}", dt, prec, it, msize)
                    it, prec, dt = get_cg_iter_time(A, r, rtol=rtol, device='cuda', method='ainv')
                    stats.put(f"PCG_Ainv_mp_{msize}", dt, prec, it, msize)

                    ic0 = ilupp.ichol0(A)
                    it, dt = get_pcg_iter_time_scipy_ichol(A, ic0, r, rtol=rtol, with_time=True)
                    stats.put(f"PCG_IC_ICHOL_Scipy_{msize}", dt, prec, it, msize)
                except RuntimeError as e:
                    logger.error(f"RuntimeError: {e}")
                except KeyboardInterrupt as e:
                    raise e

            # for sample in tqdm(data.train_dataloader(), desc="Train set"):
            #     do_inference(sample)
            for sample in tqdm(data.test_dataloader(), desc="Test set"):
                do_inference(sample)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, stop testing.")
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            raise e

        stats.rich_print()

if __name__ == "__main__":
    main()
