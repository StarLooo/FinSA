from .base_dataset_config import BaseDatasetConfig
from dataclasses import dataclass


@dataclass
class FIQAConfig(BaseDatasetConfig):
    # inherited args
    dataset_name: str = "fiqa-2018"
    dataset_path: str = f"./datasets/fiqa-2018/"
    batch_size: int = 8
