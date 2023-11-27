from .base_model_config import BaseModelConfig
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer


### todo: this is my raw you implementation, you can modify these codes as you need
@dataclass
class BertConfig(BaseModelConfig):
    # inherited args
    model_path: str = "ProsusAI/finbert"  # you can change it to local path
    model_type: PreTrainedModel = BertModel
    tokenizer_type: PreTrainedTokenizer = BertTokenizer
    padding_side: str = "right"
