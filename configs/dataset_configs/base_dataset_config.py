from dataclasses import dataclass


@dataclass
class BaseDatasetConfig:
    dataset_name: str = None
    dataset_path: str = None
    batch_size: int = None
    padding: bool = True
    truncation: bool = True
    max_length: int = 512
