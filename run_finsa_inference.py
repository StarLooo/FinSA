import time
import fire
import torch
from tqdm import tqdm
from torch import inference_mode
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from utils import *
from configs import *

ID_2_SENTIMENT = {
    "financial_phrasebank-sentences_50agree": {
        0: "negative",
        1: 'neutral',
        2: 'positive',
    },
    "fiqa-2018": {
        0: "positive",
        1: 'neutral',
        2: 'negative',
    },
    "news_with_gpt_instructions": {
        0: "negative",
        1: 'neutral',
        2: 'positive',
    },
    "twitter-financial-news-sentiment": {
        0: "negative",
        1: 'positive',
        2: 'neutral',
    }
}


### todo: you need to implement this method
@inference_mode(mode=True)
def finbert_inference(
        task_name: str,
        task_dataloader: DataLoader,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
):
    pass


@inference_mode(mode=True)
def fingpt_inference(
        task_name: str,
        task_dataloader: DataLoader,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
):
    ### preparation before inference loop
    model.eval()
    predictions, references = [], []

    ### inference loop
    for batch in tqdm(task_dataloader):
        # prepare generate inputs
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        reference = [ID_2_SENTIMENT[task_name][x.item()] for x in batch["labels"]]
        # call generate
        batch_outputs = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=8,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        # call decode and extract answer part
        generated_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        generated_texts = [generated_text.split("Answer: ")[1] for generated_text in generated_texts]
        # record predictions and references
        predictions.extend(generated_texts)
        references.extend(reference)
        # release cuda cache
        torch.cuda.empty_cache()

    return predictions, references


def main(**kwargs):
    ### extract model_name and datasets
    model_name = kwargs.pop("model_name", "fingpt-v3.2")
    datasets = kwargs.pop("datasets", "fpb,fiqa,tfns,nwgi")
    print(f"Test {model_name} on datasets: {datasets}")

    ### construct configs and update using kwargs
    if model_name.lower() == "finbert":
        model_config = BertConfig(model_name=model_name)
    else:
        model_config = LlamaFamilyModelConfig(model_name=model_name)
    dataset_configs = []
    for dataset in datasets:
        if dataset == 'fpb':
            dataset_config = FPBConfig()
        elif dataset == 'fiqa':
            dataset_config = FIQAConfig()
        elif dataset == 'tfns':
            dataset_config = TFNSConfig()
        elif dataset == 'nwgi':
            dataset_config = NWGIConfig()
        else:
            raise NotImplementedError(f"Unsupported Dataset {dataset}!")
        dataset_configs.append(dataset_config)
    update_configs([model_config] + dataset_configs, **kwargs)
    print(f"model_config:\n{model_config}")
    print(f"dataset_configs:\n{dataset_configs}")

    ### load tokenizer and model for inference
    tokenizer, model = prepare_tokenizer_and_model_inference(
        model_config=model_config
    )

    ### evaluate model on finsa tasks
    finsa_task_results = {}
    dataset_configs_tqdm = tqdm(dataset_configs)
    for dataset_config in dataset_configs_tqdm:
        task_name = dataset_config.dataset_name
        dataset_configs_tqdm.set_description(f"Evaluate on {task_name}")

        # record task start time
        start = time.time()

        # prepare inference dataloader
        if dataset_config.dataset_name == "financial_phrasebank-sentences_50agree":
            task_dataloader = prepare_fpb_inference_dataloader(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
            )
        elif dataset_config.dataset_name == "fiqa-2018":
            task_dataloader = prepare_fiqa_inference_dataloader(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
            )
        elif dataset_config.dataset_name == "news_with_gpt_instructions":
            task_dataloader = prepare_nwgi_inference_dataloader(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
            )
        elif dataset_config.dataset_name == "twitter-financial-news-sentiment":
            task_dataloader = prepare_tfns_inference_dataloader(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError

        # inference on finsa task
        if model_name.lower() == "finbert":
            predictions, references = finbert_inference(
                task_name=task_name,
                task_dataloader=task_dataloader,
                model=model,
                tokenizer=tokenizer
            )
        else:
            predictions, references = fingpt_inference(
                task_name=task_name,
                task_dataloader=task_dataloader,
                model=model,
                tokenizer=tokenizer
            )

        # compute mertic, pay attention that here we use "positive", "neural", "negative" instead of 2,1,0
        acc, micro_f1, macro_f1, weighted_f1 = compute_finsa_metric(predictions=predictions, references=references)

        # record task end time
        end = time.time()

        # record and report test result
        finsa_task_results[task_name] = {
            "acc": acc,
            "micro-F1": micro_f1,
            "macro-F1": macro_f1,
            "weighted-F1": weighted_f1,
            "predictions": predictions,
            "references": references,
            "time": end - start,
        }
        print(f"{task_name} Test Result:")
        print(f"Acc: {acc}. micro-F1: {micro_f1}. macro-F1: {macro_f1}. weighted-F1: {weighted_f1}. ")

    ### finish test
    # we can save finsa_task_results to disk if needed
    print(f"Finish Test!")


if __name__ == '__main__':
    fire.Fire(main)
