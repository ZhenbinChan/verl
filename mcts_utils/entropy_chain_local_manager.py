import time
import math
from typing import List, Dict, Any, Callable
import json
from tree_node import (
    TreeNode, 
    build_into_tree_format, 
    gather_paths, 
    print_tree
)
from evaluation import (
    check_result,
    query_local_vllm_completions_with_logprobs,
    query_local_vllm_ids_with_logprobs,
    GLM_QA_PROMPT,
    get_qwen_remote_reward_model_value,
    check_steps_format,
    query_openai_api_with_logprobs_fast,
    query_openai_api_with_logprobs_slow
)
from IPython import embed

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from utils import *

prompt_dir = "mcts_utils/prompts"
with open(os.path.join(prompt_dir, "Z3DeclarationsGeneration1.txt"), "r") as f:
    system_prompt_z3_declaration = f.read()


def repeat_lst(lst: List[Any], times: int) -> List[Any]:
    if lst is None:
        return None
    return [item for item in lst for _ in range(times)]


class EntropyGuidedChainLocalManager:
    def __init__(
        self,
        args: Dict[str, Any],
        llm: Any,
        tokenizer: Any,
        encode_fn: Callable,
        decode_fn: Callable,
        eos_tokens_set: List[int]
    ):
        """
        initialize the manager.

        :param args: the argument dictionary, containing m, n, l, etc.
        :param policy_urls: the list of policy model urls.
        :param evaluator_urls: the list of evaluator urls.
        :param eos_tokens_set: the set of end-of-sequence tokens.
        """
        self.args = args
        self.llm = llm
        self.tokenizer = tokenizer
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.eos_tokens_set = eos_tokens_set

        self.paths: Dict[str, Any] = {
            "M": args["m"],
            "N": args["n"],
            "L": args["l"],
            "T": args["t"],
            "pass_k_result": [],
            "time_use": 0,
            "tree_structures": []
        }
        self.use_api_generation = args.get("use_api_generation", False)
        self.evaluation_strategy = args.get("evaluation_strategy", "model_evaluation")

    def serialize_tree(self, node: TreeNode) -> Dict[str, Any]:
        """
        serialize the tree node, for storage.

        :param node: the TreeNode object.
        :return: the dictionary of the tree structure.
        """
        return {
            'token_list': node.token_list,
            'log_prob_list': node.log_prob_list,
            'is_end': node.is_end,
            'children': [self.serialize_tree(child) for child in node.child_nodes]
        }

    def evaluate_node(self, problem_str: str, answer_str: str, args: Dict[str, Any], node: TreeNode) -> Tuple[float, float]:
        """evaluate the score of the single node

        :return: Tuple[float, float]: (binary_score, final_score)
        """
        if node.is_end and node.finish_reason == "stop" \
            and (not args['check_step_validity'] \
                 or check_steps_format(node.total_str)[0]):
            binary_score = check_result(
                problem_str,
                node.total_str,
                answer_str,
            )[-1]
        else:
            binary_score = 0

        if args["use_pure_binary"]:
            final_score = binary_score
            return binary_score, final_score

        # Get reward model score
        value = get_qwen_remote_reward_model_value(
            urls=args["entropy_rm_urls"],
            question=problem_str,
            response=node.total_str
        )

        if args["use_pure_RM"]:
            # a, b = 0.5, -2.898
            a = args.get("a", 0.5)
            b = args.get("b", -2.898)
            print("rm_sore a", a, "b", b)
            x = a * (value - b)
            final_score = 1 / (1 + math.exp(-x))
            if self.answer_str == "" :
                binary_score = final_score
        else:
            sigmoid_value = 1 / (1 + math.exp(-value))
            final_score = binary_score + 0.5 * sigmoid_value

        return binary_score, final_score

    def evaluate_trees(self, problems: List[str], answers: List[str], args: Dict[str, Any]) -> List[float]:
        """evaluate the nodes in all trees"""

        # collect all the nodes to evaluate
        evaluation_tasks = [
            (problem_str, answer_str, args, node)
            for problem_str, answer_str, tree_list in zip(problems, answers, self.tree_lists)
            for node in tree_list
        ]

        # use thread pool to evaluate
        # with ThreadPoolExecutor(max_workers=min(32, len(evaluation_tasks))) as executor:
        with ThreadPoolExecutor(max_workers=min(8, len(evaluation_tasks))) as executor:
            results = list(executor.map(
                lambda params: self.evaluate_node(*params),
                evaluation_tasks
            ))

        # update the node scores and collect the results
        for (binary_score, final_score), (_, _, _, node) in zip(results, evaluation_tasks):
            node.binary_score = binary_score
            node.score = final_score

        pass_k_results = []
        for tree_list in self.tree_lists:
            pass_k_result = []
            for node in tree_list:
                pass_k_result.append(node.binary_score)
            pass_k_results.append(pass_k_result)

        return pass_k_results

    # 主函数，执行熵引导的链式推理
    def entropy_guided_chain(
        self,
        problems: List[str],
        answers: List[str],
        args: Dict[str, Any] = None,
        system_prompt=None,
    ) -> Dict[str, Any]:
        """
        entropy-guided chain reasoning.

        :param problem_str: the problem string.
        :param answer_str: the standard answer string.
        :return: the dictionary of the paths and results.
        """
        M = self.args["m"]
        N = self.args["n"]
        L = self.args["l"]
        T = self.args['t']
        max_length = args["generate_max_len"]

        init_inputs = []
        parsed_declarations = [] if self.evaluation_strategy == "fol" else None
        
        for problem_str in problems:
            encode_output = self.encode_fn([[problem_str]], 4096, device="cpu", system_prompt=system_prompt)
            init_input_with_template = encode_output["messages"] if self.use_api_generation else encode_output["input_ids"][0].tolist()
            init_inputs.append(init_input_with_template)

            if parsed_declarations is not None:
                declaration = get_response(problem_str, system_prompt_z3_declaration)
                parsed_declarations.append(extract_python_code(declaration))

        self.paths['init_prompt_ids_with_template'] = init_inputs
        batch_size = len(problems)
        problems = repeat_lst(problems, M)
        answers = repeat_lst(answers, M)
        init_inputs = repeat_lst(init_inputs, M)
        parsed_declarations = repeat_lst(parsed_declarations, M)
        
        time_start = time.time()

        # initialize M trees
        self.tree_lists = []
        # get the initial inference results
        for _ in range(4):
            if not self.use_api_generation:
                initial_results = query_local_vllm_ids_with_logprobs(
                    init_inputs,
                    llm=self.llm,
                    skip_special_tokens=False,
                    max_tokens=max_length,
                    stops=self.eos_tokens_set,
                    temperature=self.args["temperature"],
                    top_p=self.args["top_p"],
                    use_ray=False
                )
            else:
                initial_results = query_openai_api_with_logprobs_fast(
                    init_inputs,
                    n=1,
                    max_tokens=max_length,
                    temperature=self.args["temperature"],
                    top_p=self.args["top_p"],
                    tokenizer=self.tokenizer
                )
            # initial_results = {content_token_id_lists, content_str_lists, finish_reason_lists, token_num_lists, log_probs_lists}
            if initial_results is None or initial_results[0] is None:
                continue
            break

        for idx, (content_token_ids, _, finish_reason, _, log_probs) in enumerate(zip(*initial_results)):
            kwargs = {
                'evaluation_strategy': self.evaluation_strategy,
                'question': problems[idx],
                'declaration': parsed_declarations[idx] if parsed_declarations else None,
            }
            root_node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                decode_fn=self.decode_fn,
                token_id_list=content_token_ids,
                log_prob_list=log_probs,
                is_end=True,
                finish_reason=finish_reason,
                max_length=max_length,
                **kwargs
            )
            self.tree_lists.append([root_node])

        # iterate to expand the trees
        for iteration in range(L):
            # collect all the entropy token indices of the expandable nodes
            expansion_tasks = []
            for tree_idx, tree_list in enumerate(self.tree_lists):
                # first get the top-N nodes in each Node
                tree_entropy_tokens = []
                for node_idx, node in enumerate(tree_list):
                    if not all(node.mask):
                        if self.args['use_diverse_sampling']:
                            entropy_tokens = node.get_max_entropy_tokens(
                                top_n=N * self.args['diverse_upsampling']
                            )
                        else:
                            entropy_tokens = node.get_max_entropy_tokens(
                                top_n=N) # [token_idx, ...]
                        for token_idx in entropy_tokens:
                            entropy_value = -node.token_score[token_idx]
                            tree_entropy_tokens.append(
                                (entropy_value, tree_idx,
                                 node_idx, node, token_idx)
                            )

                # because it is the same problem, so we don't need to consider the entropy value across problems
                # select the top-N nodes as the expansion tasks
                tree_entropy_tokens.sort(reverse=True)  # sort by entropy value in descending order
                if self.args['use_diverse_sampling']:
                    # get the candidate tokens of top-(ratio*N)
                    token_indices = [token_idx for _, _,
                                     _, _, token_idx in tree_entropy_tokens]
                    scores = [entropy_value for entropy_value,
                              _, _, _, _ in tree_entropy_tokens]

                    # use diverse sampling to select the final tokens
                    selected_indices = self.select_diverse_tokens(
                        token_indices,
                        scores,
                        N,
                    )

                    # add the selected tokens to the expansion tasks
                    selected_tokens = []
                    for token_idx in selected_indices:
                        for item in tree_entropy_tokens:
                            if item[4] == token_idx:  # item[4] is token_idx
                                selected_tokens.append(item)
                                break

                    expansion_tasks.extend([
                        (tree_idx, node_idx, node, token_idx)
                        for _, tree_idx, node_idx, node, token_idx in selected_tokens
                    ])
                else:
                    expansion_tasks.extend([
                        (tree_idx, node_idx, node, token_idx)
                        for _, tree_idx, node_idx, node, token_idx in tree_entropy_tokens[:N]
                    ])

            if not expansion_tasks:
                print("no expandable nodes, terminate the iteration.")
                break

            # prepare the inference
            m_tree_top_n_inputs = []
            task_mapping = {}
            for i, (tree_idx, node_idx, node, split_idx) in enumerate(repeat_lst(expansion_tasks, T)):
                node.mask[split_idx] = True  # mark the token as masked to avoid repeated expansion in the next iterations
                prefix_ids = node.get_prefix_ids(split_idx)
                prefix_input = prefix_ids if not self.use_api_generation else \
                    [
                        {"role": "assistant", "content": self.decode_fn(prefix_ids)},
                        {"role": "user", "content": "Please continue."}
                    ]
                init_input_with_template = init_inputs[tree_idx]
                m_tree_top_n_inputs.append(init_input_with_template + prefix_input)
                task_mapping[i] = (tree_idx, node_idx, node, split_idx)

            # batch execute the inference
            if not self.use_api_generation:
                inference_results = query_local_vllm_ids_with_logprobs(
                    m_tree_top_n_inputs,
                    llm=self.llm,
                    skip_special_tokens=False,
                    max_tokens=max_length,
                    stops=self.eos_tokens_set,
                    temperature=self.args["temperature"],
                    top_p=self.args["top_p"],
                    use_ray=False
                )
            else:
                inference_results = query_openai_api_with_logprobs_fast(
                    m_tree_top_n_inputs,
                    n=1,
                    max_tokens=max_length,
                    temperature=self.args["temperature"],
                    top_p=self.args["top_p"],
                    tokenizer=self.tokenizer
                )
                if inference_results is None or inference_results[0] is None:
                    continue

            # process the results, update the tree structure
            for i, (content_token_ids, _, finish_reason, _, log_probs) in enumerate(zip(*inference_results)):
                tree_idx, node_idx, parent_node, split_idx = task_mapping[i]

                kwargs = {
                    'evaluation_strategy': self.evaluation_strategy,
                    'question': problems[tree_idx],
                    'declaration': parsed_declarations[tree_idx] if parsed_declarations else None,
                    'parent_implication': parent_node.get_prefix_implication(split_idx)
                }

                # split the current node at split_idx
                new_node = TreeNode(
                    tree_idx=tree_idx,
                    node_idx=len(self.tree_lists[tree_idx]),
                    token_id_list=content_token_ids,
                    decode_fn=self.decode_fn,
                    log_prob_list=log_probs,
                    is_end=True,
                    parent_node=parent_node,
                    parent_node_idx=node_idx,
                    parent_node_split_idx=split_idx,
                    finish_reason=finish_reason,
                    **kwargs
                )

                # build the parent-child relationship
                parent_node.add_child(new_node, split_idx)

                # add the new node to the corresponding tree list
                self.tree_lists[tree_idx].append(new_node)

        eval_time_start = time.time()

        # evaluate the results
        # pass_k_result = []
        # for tree_list in self.tree_lists:
        #     for node in tree_list:
        #         if node.is_end and node.finish_reason == "stop":
        #             response_str = node.total_str
        #             # response_str = response_str.split("<|user|>")[0]
        #             score = check_result(
        #                 problem_str,
        #                 response_str,
        #                 answer_str,
        #             )[-1]
        #             pass_k_result.append(score)
        #             node.binary_score = score
        #         else:
        #             pass_k_result.append(0)
        #             node.binary_score = 0

        # above is serial evaluation, below is parallel evaluation
        eval_time_start = time.time()
        results_per_tree = self.evaluate_trees(problems, answers, args)
        pass_k_results = [
            [
                item
                for j in range(M)
                for item in results_per_tree[i * M + j]
            ]
            for i in range(batch_size)
        ]
        self.paths['pass_k_result'] = pass_k_results
        self.paths['eval_time_use'] = time.time() - eval_time_start
        self.paths['time_use'] = time.time() - time_start

        # serialize the tree structure
        self.paths['tree_structures'] = [
            self.serialize_tree_list(tree_list) for tree_list in self.tree_lists
        ]
        path_lst = []
        for i in range(batch_size):
            root, selected_terminals = build_into_tree_format(
                self.tree_lists[i * M:(i + 1) * M],
                self.decode_fn,
                args['num_traces'],
                args["balance_ratio"],
                args["average_one_generation"],
                use_weighted_value=args["use_weighted_value"],
                use_all_terminals=args["use_all_terminals"],
                weighted_value_style=args["weighted_value_style"],
                overall_norm_style=args["overall_norm_style"],
                inner_repetition_penalty=args["inner_repetition_penalty"],
            )
            # from parallel_mcts import visualize_tree
            # visualize_tree(root)

            paths = gather_paths(
                root=root,
                selected_terminals=selected_terminals,
                pass_k=args['num_traces'],
                use_orm_reward=args['use_orm_reward'],
                use_chain_reward=args["use_chain_reward"],
                step_level_norm=args["step_level_norm"],
                use_state_value_reward=args["use_state_value_reward"],
                use_value_only=args["use_value_only"],
                average_one_generation=args["average_one_generation"],
                advantage_mix_allancestor=args["advantage_mix_allancestor"]
            )

            path_lst.append(paths)
        if args["training_type"] == "general":
            return path_lst, [sum(result_list) / len(result_list) for result_list in pass_k_results] # root.reward_raw
        return path_lst

    def serialize_tree_list(self, tree_list: List[TreeNode]) -> List[Dict[str, Any]]:
        """
        serialize the single tree list.
        """
        return [{
            'token_ids': node.token_id_list,
            'token_strs': node.token_str_list,
            'log_probs': node.log_prob_list,
            'is_end': node.is_end,
            'mask': node.mask,
            'finish_reason': node.finish_reason,
            'total_str': node.total_str,
            'parent_node_idx': node.parent_node_idx,
            'parent_node_split_idx': node.parent_node_split_idx
        } for node in tree_list]

    def process_single_item(self, item: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        """
        process the single data item.

        :param item: the data item, containing 'problem' and 'golden_answer'.
        :return: the processed paths and results.
        """
        problems = item["problem"]
        answers = item["golden_answer"]
        system_prompt = args.get("system_prompt", None)

        if args["training_type"] == "general":
            paths, raw_avg_reward = self.entropy_guided_chain(problems, answers, system_prompt=system_prompt, args=args)
        else:
            paths = self.entropy_guided_chain(problems, answers, system_prompt=system_prompt, args=args)
        
        results = {
            "problem": problems,
            "golden_answer": answers,
            "paths": paths,
            "raw_avg_reward": raw_avg_reward if args["training_type"] == "general" else None
        }
        return results

    def select_diverse_tokens(self, token_indices, scores, n):
        """
        select the most diverse n tokens from the top-k tokens

        Args:
            token_indices: the list of candidate token indices
            scores: the list of corresponding scores (entropy or probability)
            upsampling_factor: the upsampling factor

        Returns:
            selected_indices: the list of selected token indices
        """
        # n = len(token_indices) // upsampling_factor
        if n == 0:
            return token_indices

        import numpy as np

        # convert token_indices to numpy array for calculation
        tokens = np.array(token_indices)

        # initialize the selected list, select the token with the highest score first
        selected = [0]  # select the first token (the highest score)
        remaining = list(range(1, len(tokens)))

        # select the remaining tokens
        while len(selected) < n:
            max_min_dist = -float('inf')
            best_idx = -1

            # for each candidate token
            for i in remaining:
                # calculate the minimum distance (use the absolute difference of token_id as distance)
                min_dist = float('inf')
                for j in selected:
                    dist = abs(int(tokens[i]) - int(tokens[j]))
                    min_dist = min(min_dist, dist)

                # if this token can provide a larger minimum distance, select it
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        # return the selected token indices
        return [token_indices[i] for i in selected]
