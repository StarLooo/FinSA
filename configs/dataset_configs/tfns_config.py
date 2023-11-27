from .base_dataset_config import BaseDatasetConfig
from dataclasses import dataclass


### this is enough for FinGPT, but you can add needed things for FinBERT
@dataclass
class TFNSConfig(BaseDatasetConfig):
    # inherited args
    dataset_name: str = "twitter-financial-news-sentiment"
    dataset_path: str = f"./datasets/twitter-financial-news-sentiment/"
    batch_size: int = 8
