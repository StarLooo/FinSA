from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from configs import FPBConfig
from transformers import PreTrainedTokenizer, default_data_collator


def prepare_fpb_inference_dataloader(
        dataset_config: FPBConfig,
        tokenizer: PreTrainedTokenizer,
        model_name: str
) -> DataLoader:
    fpb_dataset = load_from_disk(dataset_config.dataset_path)["train"]
    fpb_inference_dataset = fpb_dataset.train_test_split(seed=42)['test']
    fpb_inference_dataset = fpb_inference_dataset.to_list()
    for sample in fpb_inference_dataset:

        if model_name.lower() != "finbert":
            sample["sentence"] = \
                "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\n" + \
                f"Input: {sample['sentence']}\n" + \
                "Answer: "
    fpb_inference_dataset = Dataset.from_list(fpb_inference_dataset)
    fpb_inference_dataset = fpb_inference_dataset.map(
        lambda sample: tokenizer(
            sample["sentence"],
            padding=dataset_config.padding,
            truncation=dataset_config.truncation,
            max_length=dataset_config.max_length,
            return_tensors="pt"
        ),
        batched=True,
        batch_size=dataset_config.batch_size,
        remove_columns=["sentence"],
    )
    # print(fpb_inference_dataset)
    fpb_inference_dataloader = DataLoader(
        fpb_inference_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    return fpb_inference_dataloader
