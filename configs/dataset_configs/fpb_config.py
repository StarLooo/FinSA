from .base_dataset_config import BaseDatasetConfig
from dataclasses import dataclass


### this is enough for FinGPT, but you can add needed things for FinBERT
@dataclass
class FPBConfig(BaseDatasetConfig):
    # inherited args
    dataset_name: str = "financial_phrasebank-sentences_50agree"
    dataset_path: str = f"./datasets/financial_phrasebank-sentences_50agree/"
    batch_size: int = 8
