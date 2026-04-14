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

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


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
