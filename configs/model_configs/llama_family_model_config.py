from .base_model_config import BaseModelConfig
from dataclasses import dataclass, field
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer

MODEL_PATHS = {  # local model paths due to missing connection to huggingface
    "llama": f"/remote-home/share/models/llama_hf/7B",
    "llama2": f"/remote-home/share/models/llama_v2_hf/7b",
    "llama2-chat": f"/remote-home/share/models/llama_v2_hf/7b-chat",
    "alpaca": f"/remote-home/xylu/hf_download/models/alpaca-7b",
    "vicuna": f"/remote-home/share/models/llama_v2_hf/vicuna-7b-v1.5",
    "fingpt-v3.2": f"/remote-home/xylu/hf_download/models/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT"
}


# temporary only support 7b size
@dataclass
class LlamaFamilyModelConfig(BaseModelConfig):
    # inherited args
    model_type: PreTrainedModel = LlamaForCausalLM
    tokenizer_type: PreTrainedTokenizer = LlamaTokenizer
    padding_side: str = "left"
    load_peft: bool = field(init=False)
    pad_token_use: str = None

    def __post_init__(self):
        model_name = self.model_name.lower()
        if model_name in ["llamav2", "llama_v2", "llama-v2", "llama2", "llama_2", "llama-2"]:
            self.load_peft = False
            self.pad_token_use = self.pad_token_use or "unk"
            if not self.model_path:
                self.model_path = MODEL_PATHS["llama2"]
        elif model_name in ["llamav2-chat", "llama_v2-chat", "llama-v2-chat", "llama2-chat", "llama_2-chat",
                            "llama-2-chat"]:
            self.load_peft = False
            self.pad_token_use = self.pad_token_use or "unk"
            if not self.model_path:
                self.model_path = MODEL_PATHS["llama2-chat"]
        elif model_name in ["llama", "llamav1", "llama_v1", "llama-v1", "llama1", "llama_1", "llama-1"]:
            self.load_peft = False
            self.pad_token_use = self.pad_token_use or "unk"
            if not self.model_path:
                self.model_path = MODEL_PATHS["llama"]
        elif model_name == "alpaca":
            self.load_peft = False
            self.pad_token_use = self.pad_token_use or "default"
            if not self.model_path:
                self.model_path = MODEL_PATHS["alpaca"]
        elif model_name == "vicuna":
            self.load_peft = False
            self.pad_token_use = self.pad_token_use or "unk"
            if not self.model_path:
                self.model_path = MODEL_PATHS["vicuna"]
        elif model_name == "fingpt-v3.2":
            self.load_peft = True
            self.pad_token_use = self.pad_token_use or "unk"
            if not self.model_path:
                self.model_path = MODEL_PATHS["llama2-chat"]
            if not self.peft_model_path:
                self.peft_model_path = MODEL_PATHS["fingpt-v3.2"]
        else:
            raise NotImplementedError
