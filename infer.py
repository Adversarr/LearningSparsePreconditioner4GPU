from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List
import hydra
from lightning import LightningDataModule
import torch
from loguru import logger
from omegaconf import DictConfig
import numpy as np
from neural_cg.utils.datamodule import FolderDataModule, MultiFolderDataModule
from neural_cg.utils.validate import (
    get_amgx_iter_time,
    get_amgxcg_iter_time,
    get_cg_iter_time,
    get_cupy_iter_time,
    get_pcg_iter_time,
    get_pcg_scaled_iter_time,
    get_pyamg_iter_time,
    get_pyamgcg_iter_time,
    to_csr_cpu,
    to_numpy,
)
from tqdm import tqdm, trange

import pandas as pd


@dataclass
class InferenceTimestat:
    all_solve_time: List[float]
    all_prec_time: List[float]
    all_iteration: List[float]
    all_matrix_size: List[int]


class Timestat:
    def __init__(self):
        self.stat_dict: Dict[str, InferenceTimestat] = {}

    def put(self, key: str, solve_time: float, prec_time: float, iteration: float, matrix_size: int):
        if key not in self.stat_dict:
            self.stat_dict[key] = InferenceTimestat([], [], [], [])
        self.stat_dict[key].all_solve_time.append(solve_time)
        self.stat_dict[key].all_prec_time.append(prec_time)
        self.stat_dict[key].all_iteration.append(iteration)
        self.stat_dict[key].all_matrix_size.append(matrix_size)

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

    def timestat_to_dataframe(self) -> pd.DataFrame:
        """
        Convert Timestat.stat_dict into a pandas DataFrame following the same structure as rich_print.

        Args:
            timestat: An instance of Timestat containing stat_dict.

        Returns:
            pd.DataFrame: A DataFrame with keys, solve times, precond times, and iteration stats.
        """
        data = []

        for key, stat in self.stat_dict.items():
            avg_solve_time = np.mean(stat.all_solve_time) * 1000
            avg_prec_time = np.mean(stat.all_prec_time) * 1000
            avg_iteration = np.mean(stat.all_iteration)

            row = {
                "Key": key,
                "Total Time (ms)": avg_solve_time + avg_prec_time,
                "Solve Time (ms)": avg_solve_time,
                "Precond Time (ms)": avg_prec_time,
                "#Iteration": avg_iteration,
            }
            data.append(row)

        # Convert to DataFrame and round numeric columns for better readability
        df = pd.DataFrame(data)
        numeric_cols = [
            "Total Time (ms)",
            "Solve Time (ms)",
            "Precond Time (ms)",
            "#Iteration",
        ]
        df[numeric_cols] = df[numeric_cols].round(4)

        return df

    def all_time_stat(self) -> pd.DataFrame:
        data = []
        for key, stat in self.stat_dict.items():
            for s, p, i, m in zip(
                stat.all_solve_time,
                stat.all_prec_time,
                stat.all_iteration,
                stat.all_matrix_size,
            ):
                row = {
                    "Key": key,
                    "Solve Time (ms)": s * 1000,
                    "Precond Time (ms)": p * 1000,
                    "#Iteration": i,
                    "Matrix Size": m,
                }
                data.append(row)
        df = pd.DataFrame(data)
        numeric_cols = [
            "Solve Time (ms)",
            "Precond Time (ms)",
            "#Iteration",
            "Matrix Size",
        ]
        df[numeric_cols] = df[numeric_cols].round(4)
        return df

def get_workspace(name):
    if name == "simple":
        from neural_cg.workspace import SimpleTrainingWorkspace

        return SimpleTrainingWorkspace, get_pcg_iter_time
    elif name == "scaled":
        from neural_cg.scaled_workspace import ScaledTrainingWorkspace

        return ScaledTrainingWorkspace, get_pcg_scaled_iter_time
    else:
        raise ValueError(f"Unknown workspace name: {name}")


@hydra.main(config_path="config", config_name="basic", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.exp_name == 'heatmultisource':
        from train_neural_pcg_heat import NeuralPCG_DataModule

        data: LightningDataModule = NeuralPCG_DataModule(
            data_config=cfg.data,
            split_config=cfg.split,
            batch_size=cfg.batch_size,
        )
        edge_input_features = 3
        node_input_features = 4
        block_size = 1
    elif cfg.exp_name == 'nif':
        from train_neural_if import NIF_DataModule

        data = NIF_DataModule(
            data_path=Path(cfg.get("data_path", "data/Random")),
            data_config=cfg.data,
            split_config=cfg.split,
            batch_size=cfg.batch_size,
        )
        edge_input_features = 1
        node_input_features = 1
        block_size = 1
    elif 'all_prefix' in cfg.data:
        data = MultiFolderDataModule(
            data_config=cfg.data,
            split_config=cfg.split,
            batch_size=cfg.batch_size,
        )
        edge_input_features = data.dataset.num_edge_features
        node_input_features = data.dataset.num_node_features
        block_size = data.dataset.block_size
    else:
        data = FolderDataModule(
            data_config=cfg.data,
            split_config=cfg.split,
            batch_size=cfg.batch_size,
        )
        edge_input_features = data.dataset.num_edge_features
        node_input_features = data.dataset.num_node_features
        block_size = data.dataset.block_size

    try:
        import pyamgx
        pyamgx.initialize()
        has_amgx = True
    except ImportError:
        logger.warning("PyAMGX is not installed. Skipping PyAMGX tests.")
        has_amgx = False

    out_dir = cfg.get("out_dir", "output")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    exp_name = cfg.exp_name
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
    rhs = cfg.get("rhs", "mask")  # use mask as rhs
    enable_cholmod = cfg.get("enable_cholmod", False)

    logger.info("Repeat each sample for {} times.".format(repeat))

    rtol: float = cfg.get("rtol", 1e-6)
    logger.info(f"Relative tolerance: {rtol}")
    logger.info(f"model.epsilon={model.epsilon:.4e}")

    with torch.no_grad():
        stats = Timestat()

        loader_name = cfg.get("dataloader", 'test')
        if loader_name == "train":
            dl = data.train_dataloader()
        elif loader_name in ["val", "test"]:
            dl = data.val_dataloader()
        elif loader_name == "infer":
            dl = data.inference_dataloader()
        else:
            raise ValueError(f"Unknown dataloader name: {loader_name}")


        # Warmup the neural network:
        iter_dl = iter(dl)
        sample = next(iter_dl)
        sample = sample.to(device)
        for _ in trange(cfg.get("warmup", 20), desc="Warmup Torch"):
            model.inference_step(sample)

        try:
            for sample in tqdm(dl, desc=f"Inference on {loader_name}"):
                sample = sample.to(device)
                mat_size = (sample.ptr[-1] * model.block_size).item()
                mask = sample.mask
                A = to_csr_cpu(sample.edge_index, sample.matrix_values, mat_size, mask)
                A_full = to_csr_cpu(
                    sample.edge_index, sample.matrix_values, mat_size, None
                )
                A_full.data.fill(1.0)

                # Preconditioner Time.
                prec = 0.0
                for _ in range(repeat):
                    _, this_prec = model.inference_step(sample)
                    prec += this_prec
                prec /= repeat

                # CG Time.
                L, _ = model.inference_step(sample)
                m = to_numpy(mask).flatten().astype(np.float64)
                if rhs == "mask" or rhs == "ones":
                    r: np.ndarray = m
                elif rhs == "random":
                    r = np.random.randn(mat_size).astype(np.float64)
                    r = r * m
                elif rhs == "neighbour":
                    r = A_full @ (1 - m) + 0.1 * m
                    r = r * m
                else:
                    raise ValueError(f"Unknown rhs type: {rhs}")
                try:
                    msize = A.shape[0]
                    for m in [
                        "none",
                        "diagonal",
                        "ainv",
                        "ic",
                    ]:  # , "fsai"]:
                        for d in ["cpu", "cuda"]:
                            print(msize, m, d)
                            it, prec, sol = get_cg_iter_time(
                                A, r, rtol=rtol, repeat=repeat, method=m, device=d
                            )
                            stats.put(f"PCG-{m}-{d}", sol, prec, it, msize)
                    print("NeuralPCG")
                    it, _, sol = pcg(
                        A, r, L, model.epsilon, rtol=rtol, device="cpu", repeat=repeat
                    )
                    print("NeuralPCG+CUDA")
                    _, _, sol_cuda = pcg(
                        A, r, L, model.epsilon, rtol=rtol, device="cuda", repeat=repeat
                    )
                    stats.put("Neural", sol, prec, it, msize)
                    stats.put("Neural+CUDA", sol_cuda, prec, it, msize)
                    #
                    # print("PyAMG+CG")
                    # it, prec, sol = get_pyamgcg_iter_time(A, r, rtol=rtol, repeat=repeat)
                    # stats.put("PyAMG+CG", sol, prec, it, msize)
                    # print("AMG+CG")
                    # it, prec, sol = get_amgxcg_iter_time(A, r, rtol=rtol, repeat=repeat)
                    # stats.put("AMGX+CG", sol, prec, it, msize)
                    # print("PyAMG")
                    # it, prec, sol = get_pyamg_iter_time(A, r, rtol=rtol, repeat=repeat)
                    # stats.put("PyAMG", sol, prec, it, msize)
                    # print("AMG")
                    # it, prec, sol = get_amgx_iter_time(A, r, rtol=rtol, repeat=repeat)
                    # stats.put("AMGX", sol, prec, it, msize)
                    #
                    # print("cupycg")
                    # it, prec, sol = get_cupy_iter_time(A, r, rtol=rtol, repeat=repeat)
                    # stats.put("cupycg", sol, prec, it, msize)
                    #

                    if enable_cholmod:
                        from pymathprim.linalg.cholmod import chol
                        b, x = np.ones_like(r), np.zeros_like(r)
                        begin_time = time.time()
                        llt = chol(A)
                        prec = time.time() - begin_time
                        begin_time = time.time()
                        llt.solve(b, x)
                        solve_time = time.time() - begin_time
                        stats.put("Cholmod", solve_time, prec, 1, msize)
                except KeyboardInterrupt as e:
                    raise e
                except RuntimeError as e:
                    logger.error(f"RuntimeError: {e}")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, stop testing.")
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            exit(-1)

        stats.rich_print()

        df = stats.timestat_to_dataframe()
        log_rtol = -int(math.log10(rtol))
        prefix = cfg.get('infer_prefix', '')
        fname = out_dir / f"infer_{prefix}{exp_name}_{log_rtol}.csv"
        df.to_csv(fname, index=False)
        logger.info("Inference statistics saved to {}".format(fname))

        df = stats.all_time_stat()
        all_fname = out_dir / f"all_infer_{prefix}{exp_name}_{log_rtol}.csv"
        df.to_csv(all_fname, index=False)
        logger.info("All inference statistics saved to {}".format(all_fname))


if __name__ == "__main__":
    main()
