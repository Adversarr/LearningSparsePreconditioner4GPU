from typing import List
import lightning as L
from loguru import logger
from neural_cg.data import FolderDataset, MultiFolderDataset
import numpy as np
from sklearn.model_selection import train_test_split

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


class SubsetAdaptor(Dataset):
    def __init__(self, dataset: Dataset, indices: list):
        super().__init__()
        self.dataset = dataset
        self.inds = indices

    def len(self):
        return len(self.inds)

    def get(self, idx):
        return self.dataset.get(self.inds[idx])


def split_dataset(dataset_len, train_ratio):
    train_idx, val_idx = train_test_split(range(dataset_len), train_size=train_ratio, random_state=42, shuffle=True)
    return list(train_idx), list(val_idx)


class FolderDataModule(L.LightningDataModule):
    def __init__(self, data_config: dict, split_config, batch_size):
        super().__init__()
        self.dataset = FolderDataset(**data_config)
        train_ratio = split_config["train"]
        self.batch_size = batch_size

        try:
            # split the dataset using Subset
            logger.info(f"Splitting dataset into train/val/test with ratios: {train_ratio}, {1 - train_ratio}")
            self.train_idx, self.val_idx = split_dataset(
                len(self.dataset),
                train_ratio,
            )
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")

    def inference_dataloader(self):
        # Use the entire dataset for inference
        return DataLoader(self.dataset, batch_size=1, shuffle=False)

    def train_dataloader(self):
        train_subset: Dataset = SubsetAdaptor(self.dataset, self.train_idx)
        return DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_subset: Dataset = SubsetAdaptor(self.dataset, self.val_idx)
        return DataLoader(val_subset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


class MultiFolderDataModule(L.LightningDataModule):
    def __init__(self, data_config, split_config, batch_size):
        super().__init__()
        self.dataset = MultiFolderDataset(**data_config) # type: ignore
        train_ratio = split_config["train"]
        self.batch_size = batch_size

        # split the dataset using Subset
        logger.info(f"Splitting dataset into train/val/test with ratios: {train_ratio}, {1 - train_ratio}")
        self.train_idx, self.val_idx = split_dataset(
            len(self.dataset),
            train_ratio,
        )

    def inference_dataloader(self):
        # Use the entire dataset for inference
        return DataLoader(self.dataset, batch_size=1, shuffle=False)

    def train_dataloader(self):
        train_subset: Dataset = SubsetAdaptor(self.dataset, self.train_idx)
        return DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_subset: Dataset = SubsetAdaptor(self.dataset, self.val_idx)
        return DataLoader(val_subset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()
