from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class BaseModelConfig:
    model_name: str = None
    load_peft: bool = False
    model_path: str = None
    model_type: PreTrainedModel = None
    peft_model_path: str = None
    tokenizer_type: PreTrainedTokenizer = None
    pad_token_use: str = "default"  # unk, bos, eos, additional
    padding_side: str = None  # left, right
    load_in_8bit: bool = False  # do not need when inference on RTX3090
