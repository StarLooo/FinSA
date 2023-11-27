import torch
from peft import PeftModel
from accelerate.utils import load_and_quantize_model, BnbQuantizationConfig
from configs import BaseModelConfig


def prepare_tokenizer_and_model_inference(
        model_config: BaseModelConfig = None
):
    if model_config.model_name.lower() == "finbert":
        return finbert_prepare_tokenizer_and_model_inference(model_config=model_config)
    else:
        return fingpt_prepare_tokenizer_and_model_inference(model_config=model_config)


def fingpt_prepare_tokenizer_and_model_inference(
        model_config: BaseModelConfig = None
):
    ### load tokenizer
    tokenizer = model_config.tokenizer_type.from_pretrained(
        model_config.model_path,
        padding_side=model_config.padding_side,
    )

    ### get pad_token setting from model_config and adjust tokenizer
    if model_config.pad_token_use == 'default':
        pad_token_id = tokenizer.pad_token_id
    elif model_config.pad_token_use == 'additional':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pad_token_id = tokenizer.pad_token_id
    elif model_config.pad_token_use == 'bos':
        pad_token_id = tokenizer.pad_token_id = tokenizer.bos_token_id
        tokenizer.pad_token = tokenizer.bos_token
    elif model_config.pad_token_use == 'eos':
        pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_config.pad_token_use == 'unk':
        pad_token_id = tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError("pad_token_use must select in ['default', 'additional', 'bos', 'eos', 'unk']!")

    ### load model and adjust embeddings if use additional pad_token
    model = model_config.model_type.from_pretrained(
        model_config.model_path,
        load_in_8bit=model_config.load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=pad_token_id
    )
    if model_config.pad_token_use == 'additional':
        model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + 1, pad_to_multiple_of=64)

    # load peft model if needed
    if model_config.load_peft:
        model = PeftModel.from_pretrained(
            model,
            model_config.peft_model_path,
            torch_dtype=torch.float16,
        )
        if model.peft_type == 'LORA':  # merge for faster inference
            if model_config.load_in_8bit:
                print(
                    f"Pay attention that we need merge and unload LoRA module to the base model for faster inference. "
                    f"But merge_and_unload() method doesn't support base model loaded in int8 dtype. "
                    f"Thus the generation speed may be very slow. "
                    f"The recommended approach is to load the base model in float16, merge the LoRA weights, "
                    f"save the new model somewhere (optional) and load/quantize it into 8bit. "
                    f"We use utils in accelerate to perform the above steps. "
                    f"However the correctness of this approach still requires careful verification!"
                )
                quantization_config = BnbQuantizationConfig(load_in_8bit=True)
                model = load_and_quantize_model(model, bnb_quantization_config=quantization_config)
            model = model.merge_and_unload()

    return tokenizer, model


### todo: you need to implement this, but I think it's very similart to implementation in fingpt_prepare_tokenizer_and_model_inference()
def finbert_prepare_tokenizer_and_model_inference(
        model_config: BaseModelConfig = None
):
    pass
