from .base_dataset_config import BaseDatasetConfig
from dataclasses import dataclass


### this is enough for FinGPT, but you can add needed things for FinBERT
@dataclass
class NWGIConfig(BaseDatasetConfig):
    # inherited args
    dataset_name: str = "news_with_gpt_instructions"
    dataset_path: str = f"./datasets/news_with_gpt_instructions/"
    batch_size: int = 8
