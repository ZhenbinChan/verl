from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
