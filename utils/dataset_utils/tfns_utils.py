from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from configs import TFNSConfig
from transformers import PreTrainedTokenizer, default_data_collator


def prepare_tfns_inference_dataloader(
        dataset_config: TFNSConfig,
        tokenizer: PreTrainedTokenizer,
) -> DataLoader:
    tfns_inference_dataset = load_from_disk(dataset_config.dataset_path)["validation"]
    tfns_inference_dataset = tfns_inference_dataset.to_list()
    for sample in tfns_inference_dataset:
        sample["text"] = \
            "Instruction: What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.\n" + \
            f"Input: {sample['text']}\n" + \
            "Answer: "
    tfns_inference_dataset = Dataset.from_list(tfns_inference_dataset)
    tfns_inference_dataset = tfns_inference_dataset.map(
        lambda sample: tokenizer(
            sample["text"],
            padding=dataset_config.padding,
            truncation=dataset_config.truncation,
            max_length=dataset_config.max_length,
            return_tensors="pt"
        ),
        batched=True,
        batch_size=dataset_config.batch_size,
        remove_columns=["text"],
    )
    # print(tfns_inference_dataset)
    tfns_inference_dataloader = DataLoader(
        tfns_inference_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    return tfns_inference_dataloader
