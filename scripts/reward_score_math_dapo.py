from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import gsm8k


def compute_score(data_source, response, ground_truth, extra_info=None):
    if data_source == "openai/gsm8k":
        # GSM8K greedy eval may omit the strict "#### <ans>" format.
        # Use flexible extraction to avoid undercounting correct numeric answers.
        result = gsm8k.compute_score(response, ground_truth, method="flexible")
    else:
        result = default_compute_score(data_source, response, ground_truth, extra_info)
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)
