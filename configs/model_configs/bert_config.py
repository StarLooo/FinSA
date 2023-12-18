from .base_model_config import BaseModelConfig
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer, BertForSequenceClassification, BertTokenizer


@dataclass
class BertConfig(BaseModelConfig):
    # inherited args
    model_path: str = f"/root/duruibin/finBERT/models/sentiment/FinBERT"
    model_type: PreTrainedModel = BertForSequenceClassification
    tokenizer_type: PreTrainedTokenizer = BertTokenizer
    padding_side: str = "right"
    # new args
    output_mode: str = "classification"
    do_lower_case: bool = True
