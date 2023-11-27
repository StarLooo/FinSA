from sklearn.metrics import accuracy_score, f1_score


def postprocess(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'


def compute_finsa_metric(
        predictions: list[str],
        references: list[str]
):
    assert len(predictions) == len(references), \
        f"predictions length ({len(predictions)}) equal to references length ({len(references)})!"
    postprocessed_predictions = [postprocess(x) for x in predictions]
    postprocessed_references = [postprocess(x) for x in references]
    acc = accuracy_score(postprocessed_references, postprocessed_predictions)
    micro_f1 = f1_score(postprocessed_references, postprocessed_predictions, average="micro")
    macro_f1 = f1_score(postprocessed_references, postprocessed_predictions, average="macro")
    weighted_f1 = f1_score(postprocessed_references, postprocessed_predictions, average="weighted")
    return acc, micro_f1, macro_f1, weighted_f1
