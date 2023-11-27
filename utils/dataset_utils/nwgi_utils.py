from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from configs import NWGIConfig
from transformers import PreTrainedTokenizer, default_data_collator

_label_mapping = {
    'strong negative': 0,
    'moderately negative': 0,
    'mildly negative': 1,
    'strong positive': 2,
    'moderately positive': 2,
    'mildly positive': 1,
    'neutral': 1,
}


def prepare_nwgi_inference_dataloader(
        dataset_config: NWGIConfig,
        tokenizer: PreTrainedTokenizer,
) -> DataLoader:
    nwgi_inference_dataset = load_from_disk(dataset_config.dataset_path)["test"]
    nwgi_inference_dataset = nwgi_inference_dataset.to_list()
    for sample in nwgi_inference_dataset:
        sample["label"] = _label_mapping[sample["label"]]
        sample["news"] = \
            "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\n" + \
            f"Input: {sample['news']}\n" + \
            "Answer: "
    nwgi_inference_dataset = Dataset.from_list(nwgi_inference_dataset)
    nwgi_inference_dataset = nwgi_inference_dataset.map(
        lambda sample: tokenizer(
            sample["news"],
            padding=dataset_config.padding,
            truncation=dataset_config.truncation,
            max_length=dataset_config.max_length,
            return_tensors="pt"
        ),
        batched=True,
        batch_size=dataset_config.batch_size,
        remove_columns=["news", "prompt", "out", "prompt_tokens", "completion_tokens", "total_tokens"],
    )
    # print(nwgi_inference_dataset)
    nwgi_inference_dataloader = DataLoader(
        nwgi_inference_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    return nwgi_inference_dataloader
