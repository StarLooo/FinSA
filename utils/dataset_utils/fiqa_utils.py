import datasets
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from configs import FIQAConfig
from transformers import PreTrainedTokenizer, default_data_collator


def prepare_fiqa_inference_dataloader(
        dataset_config: FIQAConfig,
        tokenizer: PreTrainedTokenizer,
        model_name: str
) -> DataLoader:
    fiqa_datasets = load_from_disk(dataset_config.dataset_path)
    fiqa_datasets = datasets.concatenate_datasets(
        [fiqa_datasets["train"], fiqa_datasets["validation"], fiqa_datasets["test"]])
    fiqa_inference_dataset = fiqa_datasets.train_test_split(0.226, seed=42)['test']
    fiqa_inference_dataset = fiqa_inference_dataset.to_list()
    for sample in fiqa_inference_dataset:

        if model_name.lower() != "finbert":

            if sample["format"] == "post":
                sample["sentence"] = \
                    "Instruction: What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.\n" + \
                    f"Input: {sample['sentence']}\n" + \
                    "Answer: "
            else:
                sample["sentence"] = \
                    "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\n" + \
                    f"Input: {sample['sentence']}\n" + \
                    "Answer: "
    fiqa_inference_dataset = Dataset.from_list(fiqa_inference_dataset)
    fiqa_inference_dataset = fiqa_inference_dataset.map(
        lambda sample: tokenizer(
            sample["sentence"],
            padding=dataset_config.padding,
            truncation=dataset_config.truncation,
            max_length=dataset_config.max_length,
            return_tensors="pt"
        ),
        batched=True,
        batch_size=dataset_config.batch_size,
        remove_columns=["sentence", "snippets", "target", "sentiment_score", "aspects", "format"],
    )
    # print(fiqa_inference_dataset)
    fiqa_inference_dataloader = DataLoader(
        fiqa_inference_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
    )
    return fiqa_inference_dataloader
