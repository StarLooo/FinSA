from .base_model_config import BaseModelConfig
from dataclasses import dataclass, field
from transformers import AutoModel,  AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

@dataclass
class ChatGlmModelConfig(BaseModelConfig):
    # inherited args
    model_type: AutoModel = AutoModel
    tokenizer_type: AutoTokenizer = AutoTokenizer
    padding_side: str = "left"

    def __post_init__(self):
        self.load_peft = True
        self.pad_token_use = self.pad_token_use or "default"
        self.model_path = "THUDM/chatglm2-6b"
        self.peft_model_path = "oliverwang15/FinGPT_v31_ChatGLM2_Sentiment_Instruction_LoRA_FT"