from .base_model_config import BaseModelConfig
from dataclasses import dataclass
from transformers import (PreTrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer,
                          AutoModelForSequenceClassification, AutoTokenizer)


### todo: this is my raw you implementation, you can modify these codes as you need
@dataclass
class BertConfig(BaseModelConfig):
    # inherited args
    model_path: str = f"/root/duruibin/finBERT/models/sentiment/FinBERT"  # you can change it to local path
    # model_type: PreTrainedModel = BertModel
    model_type: PreTrainedModel = AutoModelForSequenceClassification
    # tokenizer_type: PreTrainedTokenizer = BertTokenizer
    tokenizer_type: PreTrainedTokenizer = AutoTokenizer
    padding_side: str = "right"

    output_mode: str = "classification"
    do_lower_case: bool = True
