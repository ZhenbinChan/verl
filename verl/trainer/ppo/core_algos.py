# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F

# Policy loss function type
PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig],  # config
        torch.Tensor | None,  # rollout_is_weights
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name: str) -> PolicyLossFn:
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    if name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_entropy_reinforce_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
):
    """TreeRL-aligned dense reward as advantage.

    In TreeRL's reinforce path, token-level rewards are used directly in the
    clipped policy objective. We mimic that behavior by treating masked
    token-level rewards as both advantages and returns.
    """
    advantages = token_level_rewards * response_mask
    return advantages, advantages


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def compute_grpo_prm_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    score_idx: torch.Tensor,
    reward_mask: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO with Process Reward Model (PRM).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            prompt indices for grouping rollouts
        score_idx: `(torch.Tensor)`
            indices of step ending tokens, padded with -1
            shape: (bs, max_steps)
        reward_mask: `(torch.Tensor)`
            mask for valid steps (1 for valid, 0 for padding)
            shape: (bs, max_steps)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # print(score_idx.shape)
    # print(token_level_rewards.shape)
    bsz, response_length = token_level_rewards.shape
    max_steps = score_idx.shape[1]
    
    # Initialize token-level advantages as zeros
    token_advantages = torch.zeros_like(token_level_rewards)
    
    with torch.no_grad():
        # Process each step position
        for step in range(max_steps):
            # Collect step rewards for current step across all samples
            step_rewards = []
            valid_samples = []
            
            for i in range(bsz):
                if reward_mask[i, step] > 0 and score_idx[i, step] >= 0:
                    step_pos = score_idx[i, step]
                    step_reward = token_level_rewards[i, step_pos]
                    step_rewards.append(step_reward)
                    valid_samples.append(i)
            
            if len(step_rewards) == 0:
                continue
                
            step_rewards = torch.stack(step_rewards)
            
            # Group by prompt index for current step
            id2score = defaultdict(list)
            id2mean = {}
            id2std = {}
            
            for idx, sample_i in enumerate(valid_samples):
                prompt_idx = index[sample_i]
                id2score[prompt_idx].append(step_rewards[idx])
            
            # Compute mean and std for each prompt group
            for prompt_idx in id2score:
                if len(id2score[prompt_idx]) == 1:
                    id2mean[prompt_idx] = torch.tensor(0.0)
                    id2std[prompt_idx] = torch.tensor(1.0)
                elif len(id2score[prompt_idx]) > 1:
                    id2mean[prompt_idx] = torch.mean(torch.stack(id2score[prompt_idx]))
                    id2std[prompt_idx] = torch.std(torch.stack(id2score[prompt_idx]))
                else:
                    raise ValueError(f"no score in prompt index: {prompt_idx}")
            
            # Compute step-level advantages and assign to specific token positions
            for idx, sample_i in enumerate(valid_samples):
                prompt_idx = index[sample_i]
                step_pos = score_idx[sample_i, step]
                
                if norm_adv_by_std_in_grpo:
                    step_advantage = (step_rewards[idx] - id2mean[prompt_idx]) / (id2std[prompt_idx] + epsilon)
                else:
                    step_advantage = step_rewards[idx] - id2mean[prompt_idx]
                
                # Only assign advantage to the specific step ending token
                token_advantages[sample_i, step_pos] = step_advantage
        
        # Apply response mask
        token_advantages = token_advantages * response_mask

    return token_advantages, token_advantages

def compute_tree_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    score_idx: torch.Tensor,
    reward_mask: torch.Tensor,
    token_level_values: torch.Tensor,
    token_level_q_values: torch.Tensor,
    gamma: float = 0.99,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """Combine step-level (Q-V) advantage with GRPO-style outcome advantage for tree rollouts.

    Args:
        token_level_rewards: step rewards per token (shape: bs x resp_len)
        response_mask: mask over valid response tokens (bs x resp_len)
        index: prompt indices for grouping rollouts (bs,)
        score_idx: step end token positions (bs x max_steps, padded with -1)
        reward_mask: mask for valid steps (bs x max_steps)
        token_level_values: state-value per token (bs x resp_len)
        token_level_q_values: Q/return per token (bs x resp_len)
        gamma: discount factor (currently unused here; provided for completeness)
    Returns:
        advantages: (bs x resp_len)
        returns: (bs x resp_len)
    """
    bsz, resp_len = token_level_rewards.shape

    # -------- Step-level Advantage: A = Q - V, broadcast to each step span --------
    step_adv = torch.zeros_like(token_level_rewards)
    for i in range(bsz):
        last_end = -1
        for j in range(score_idx.size(1)):
            if reward_mask[i, j] <= 0:
                continue
            end_pos = score_idx[i, j]
            if end_pos < 0 or end_pos >= resp_len:
                continue
            start_pos = last_end + 1
            start_pos = max(0, start_pos)
            start_pos = min(start_pos, resp_len - 1)
            # advantage at the step end token
            step_a = token_level_q_values[i, end_pos] - token_level_values[i, end_pos]
            # broadcast to tokens within this step span
            step_adv[i, start_pos : end_pos + 1] = step_a
            last_end = end_pos
    step_adv = step_adv * response_mask

    # -------- Traditional GRPO Advantage using final step reward per trajectory --------
    outcome_reward_mat = torch.zeros_like(token_level_rewards)
    for i in range(bsz):
        # find last valid step
        valid_positions = [score_idx[i, j].item() for j in range(score_idx.size(1)) if reward_mask[i, j] > 0 and score_idx[i, j] >= 0]
        if not valid_positions:
            continue
        last_pos = max(valid_positions)
        last_pos = min(last_pos, resp_len - 1)
        outcome_reward_mat[i, last_pos] = token_level_rewards[i, last_pos]

    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=outcome_reward_mat,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    # -------- Combine --------
    combined_adv = step_adv + grpo_adv
    combined_adv = combined_adv * response_mask

    return combined_adv, combined_adv


def compute_mcts_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    score_idx: torch.Tensor,
    reward_mask: torch.Tensor,
    step_correctness_scores: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """MCTS-specific advantage for parallel_mcts sampling strategy.

    Combines two signals:
    1. Step-level: A_step = PRM - V_baseline
       V_baseline = correct_terminal_in_subtree / terminal_in_subtree per step node
    2. GRPO Outcome: ORM (0/1) at final step, normalized within prompt groups

    Args:
        token_level_rewards: step rewards per token [B, L_resp], PRM format
        response_mask: mask over valid response tokens [B, L_resp]
        index: prompt indices for grouping [B]
        score_idx: step end token positions [B, max_steps]
        reward_mask: mask for valid steps [B, max_steps]
        step_correctness_scores: V per step [B, max_steps]
            = correct_terminal_in_subtree / terminal_in_subtree
        epsilon: numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std (GRPO style)

    Returns:
        advantages, returns: both [B, L_resp]
    """
    bsz, resp_len = token_level_rewards.shape
    max_steps = score_idx.size(1)

    # -------- Step-level Advantage: A = PRM - V --------
    step_adv = torch.zeros_like(token_level_rewards)
    for i in range(bsz):
        last_end = -1
        for j in range(max_steps):
            if reward_mask[i, j] <= 0:
                continue
            end_pos = score_idx[i, j]
            if end_pos < 0 or end_pos >= resp_len:
                continue
            start_pos = max(0, last_end + 1)
            end_pos = min(end_pos, resp_len - 1)
            # A_step = PRM - V(correct/terminal)
            step_a = token_level_rewards[i, end_pos] - step_correctness_scores[i, j]
            step_adv[i, start_pos : end_pos + 1] = step_a
            last_end = end_pos
    step_adv = step_adv * response_mask

    # -------- GRPO Outcome Advantage using final step reward --------
    outcome_reward_mat = torch.zeros_like(token_level_rewards)
    for i in range(bsz):
        valid_positions = [
            score_idx[i, j].item()
            for j in range(max_steps)
            if reward_mask[i, j] > 0 and score_idx[i, j] >= 0
        ]
        if not valid_positions:
            continue
        last_pos = min(max(valid_positions), resp_len - 1)
        outcome_reward_mat[i, last_pos] = token_level_rewards[i, last_pos]

    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=outcome_reward_mat,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    # -------- Combine --------
    combined_adv = step_adv + grpo_adv
    combined_adv = combined_adv * response_mask
    return combined_adv, combined_adv


def compute_step_treerl_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    score_idx: torch.Tensor,
    reward_mask: torch.Tensor,
    step_correctness_scores: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """Step-level advantage for step_treerl sampling strategy.

    A_step = PRM - V_baseline per step, broadcast to all tokens within the step.
    V_baseline = correct_terminal_in_subtree / terminal_in_subtree per step node.
    Optionally applies GRPO-style group normalization.

    Args:
        token_level_rewards: step rewards per token [B, L_resp]
        response_mask: mask over valid response tokens [B, L_resp]
        index: prompt indices for grouping [B]
        score_idx: step end token positions [B, max_steps], padded with -1
        reward_mask: mask for valid steps [B, max_steps]
        step_correctness_scores: V per step [B, max_steps]
        epsilon: numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std (GRPO style)

    Returns:
        advantages, returns: both [B, L_resp]
    """
    bsz, resp_len = token_level_rewards.shape
    max_steps = score_idx.size(1)

    # Step-level: A = PRM - V(correct/terminal), broadcast to step span
    step_adv = torch.zeros_like(token_level_rewards)
    for i in range(bsz):
        last_end = -1
        for j in range(max_steps):
            if reward_mask[i, j] <= 0:
                continue
            end_pos = score_idx[i, j]
            if end_pos < 0 or end_pos >= resp_len:
                continue
            start_pos = max(0, last_end + 1)
            end_pos = min(end_pos, resp_len - 1)
            step_a = token_level_rewards[i, end_pos] - step_correctness_scores[i, j]
            step_adv[i, start_pos : end_pos + 1] = step_a
            last_end = end_pos
    step_adv = step_adv * response_mask

    # GRPO-style group normalization
    if norm_adv_by_std_in_grpo:
        with torch.no_grad():
            id2mean = {}
            id2std = {}
            for i in range(bsz):
                pid = index[i]
                if pid not in id2mean:
                    mask = index == pid
                    group_adv = step_adv[mask]
                    group_scores = group_adv.sum(dim=-1)
                    id2mean[pid] = group_scores.mean()
                    id2std[pid] = group_scores.std() if len(group_scores) > 1 else torch.tensor(1.0, device=step_adv.device)
            for i in range(bsz):
                pid = index[i]
                step_adv[i] = (step_adv[i] - id2mean[pid]) / (id2std[pid] + epsilon)
    else:
        with torch.no_grad():
            id2mean = {}
            for i in range(bsz):
                pid = index[i]
                if pid not in id2mean:
                    mask = index == pid
                    group_adv = step_adv[mask]
                    id2mean[pid] = group_adv.sum(dim=-1).mean()
            for i in range(bsz):
                pid = index[i]
                step_adv[i] = step_adv[i] - id2mean[pid]

    step_adv = step_adv * response_mask
    return step_adv, step_adv


def compute_ig_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    score_idx: torch.Tensor,
    reward_mask: torch.Tensor,
    step_correctness_scores: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """Step-level advantage for information_gain sampling strategy.

    Identical formula to compute_step_treerl_advantage:
    A_step = PRM - V_baseline per step, broadcast to all tokens within the step.
    Separate function for potential future divergence.

    Args:
        token_level_rewards: step rewards per token [B, L_resp]
        response_mask: mask over valid response tokens [B, L_resp]
        index: prompt indices for grouping [B]
        score_idx: step end token positions [B, max_steps], padded with -1
        reward_mask: mask for valid steps [B, max_steps]
        step_correctness_scores: V per step [B, max_steps]
        epsilon: numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std (GRPO style)

    Returns:
        advantages, returns: both [B, L_resp]
    """
    return compute_step_treerl_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score_idx=score_idx,
        reward_mask=reward_mask,
        step_correctness_scores=step_correctness_scores,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


#def compute_policy_loss(
#    old_log_prob,
#    log_prob,
#    advantages,
#    response_mask,
#    cliprange=None,
#    cliprange_low=None,
#    cliprange_high=None,
#    clip_ratio_c=3.0,
#    loss_agg_mode="token-mean",
#):
#    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
#    Args:
#        old_log_prob: `(torch.Tensor)`
#            shape: (bs, response_length)
#        log_prob: `(torch.Tensor)`
#            shape: (bs, response_length)
#        advantages: `(torch.Tensor)`
#            shape: (bs, response_length)
#        response_mask: `(torch.Tensor)`
#            shape: (bs, response_length)
#        cliprange: (float)
#            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
#        cliprange_low: (float)
#            The lower clip range used in PPO.
#        cliprange_high: (float)
#            The higher clip range used in PPO.
#        clip_ratio_c: (float) default: 3.0
#            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
#        loss_agg_mode: (str) choices: "token-mean" /
#                                      "seq-mean-token-sum" /
#                                      "seq-mean-token-mean" /
#                                      "seq-mean-token-sum-norm" /
#            "token-mean" is the default behavior
#
#    Returns:
#        pg_loss: `a scalar torch.Tensor`
#            policy gradient loss computed via PPO
#        pg_clipfrac: (float)
#            the fraction of policy gradient loss being clipped
#        ppo_kl: (float)
#            the estimated KL divergence between the latest updating policy and the old sampling policy
#        pg_clipfrac_lower: (float)
#            the fraction of policy gradient loss being clipped when the advantage is negative
#    """
#    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
#
#    negative_approx_kl = log_prob - old_log_prob
#    ratio = torch.exp(negative_approx_kl)
#    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
#
#    pg_losses1 = -advantages * ratio
#    if cliprange_low is None:
#        cliprange_low = cliprange
#    if cliprange_high is None:
#        cliprange_high = cliprange
#    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
#    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
#    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
#
#    pg_losses3 = -advantages * clip_ratio_c
#    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
#    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
#
#    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
#    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
#
#    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    avg_ratio = verl_F.masked_mean(ratio, eos_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl, avg_ratio

def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


# ============ 以下为新增的 policy loss 函数 (来自最新版本 verl) ============


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO (DAPO Clip-Higher).

    This version supports asymmetric clipping with clip_ratio_low and clip_ratio_high.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config (Optional[DictConfig]):
            Config with clip_ratio, clip_ratio_low, clip_ratio_high.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    clip_ratio = config.clip_ratio if config is not None and hasattr(config, 'clip_ratio') else 0.2
    clip_ratio_low = config.clip_ratio_low if config is not None and hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config is not None and hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.clip_ratio_c if config is not None and hasattr(config, 'clip_ratio_c') else 3.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.
    CISPO: Clipped Implicit Self-Play Optimization

    See https://arxiv.org/pdf/2506.13585 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config (Optional[DictConfig]):
            Config with clip_ratio_low, clip_ratio_high.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    clip_ratio = config.clip_ratio if config is not None and hasattr(config, 'clip_ratio') else 0.2
    clip_ratio_low = config.clip_ratio_low if config is not None and hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config is not None and hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    pg_losses = -clipped_ratio_sg * advantages * log_prob

    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config (Optional[DictConfig]):
            Config with clip_ratio, clip_ratio_low, clip_ratio_high, clip_ratio_c.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    clip_ratio = config.clip_ratio if config is not None and hasattr(config, 'clip_ratio') else 0.2
    clip_ratio_low = config.clip_ratio_low if config is not None and hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config is not None and hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.clip_ratio_c if config is not None and hasattr(config, 'clip_ratio_c') else 20.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_ratio_high
    valid_negative_mask = (prob - old_prob) >= -clip_ratio_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config (Optional[DictConfig]):
            Config with clip_ratio, clip_ratio_low, clip_ratio_high, clip_ratio_c.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    clip_ratio = config.clip_ratio if config is not None and hasattr(config, 'clip_ratio') else 0.2
    clip_ratio_low = config.clip_ratio_low if config is not None and hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config is not None and hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.clip_ratio_c if config is not None and hasattr(config, 'clip_ratio_c') else 20.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_ratio_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_ratio_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, recommended to use "seq-mean-token-mean".
        config (Optional[DictConfig]):
            Config with clip_ratio_low, clip_ratio_high.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    clip_ratio = config.clip_ratio if config is not None and hasattr(config, 'clip_ratio') else 0.2
    clip_ratio_low = config.clip_ratio_low if config is not None and hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config is not None and hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)

    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, recommended to use "seq-mean-token-mean".
        config (Optional[DictConfig]):
            Config with tau_pos, tau_neg.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    tau_pos = config.tau_pos if config is not None and hasattr(config, 'tau_pos') else 1.0
    tau_neg = config.tau_neg if config is not None and hasattr(config, 'tau_neg') else 1.0

    tau_pos = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,
        other=tau_neg,
    )

    gates = gate_function(ratio, taus)

    pg_losses = -gates * advantages

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the policy gradient loss for GPG (Guided Policy Gradient).

    Adapted from https://github.com/AMAP-ML/GPG

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config (Optional[DictConfig]):
            Config (not used in this loss).
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    pg_losses = -log_prob * advantages

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, {}


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO (Geo-Mean Policy Optimization).

    Adapted from paper https://arxiv.org/abs/2507.20673

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Not used for geo_mean.
        config (Optional[DictConfig]):
            Config with clip_ratio_low, clip_ratio_high.
        rollout_is_weights (torch.Tensor | None):
            Importance sampling weights for rollout correction.
    """
    clip_ratio = config.clip_ratio if config is not None and hasattr(config, 'clip_ratio') else 0.2
    clip_ratio_low = config.clip_ratio_low if config is not None and hasattr(config, 'clip_ratio_low') and config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config is not None and hasattr(config, 'clip_ratio_high') and config.clip_ratio_high is not None else clip_ratio

    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -clip_ratio_low, clip_ratio_high)
    negative_approx_kl_min = torch.minimum(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    if rollout_is_weights is not None:
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics
