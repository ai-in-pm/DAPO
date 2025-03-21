# DAPO Implementation Guide

This document provides a detailed technical overview of the Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) algorithm implementation. It explains the key components, design decisions, and best practices used in this implementation.

## Algorithm Overview

DAPO is designed to fine-tune language models for complex reasoning tasks, addressing key challenges in existing Reinforcement Learning from Human Feedback (RLHF) approaches. The DAPO algorithm introduces several innovations:

### 1. Group Relative Policy Optimization (GRPO)

Unlike standard PPO which considers each prompt-response independently, DAPO samples G responses per prompt and normalizes rewards relative to the group.

**Implementation**: `models/dapo.py::DAPOAgent.compute_advantages`

```python
# Compute normalized advantages within each group
for i in range(0, len(prompt_ids), group_size):
    group_end = min(i + group_size, len(prompt_ids))
    group_rewards = rewards[i:group_end]
    
    # Normalize rewards within the group
    reward_mean = group_rewards.mean()
    reward_std = group_rewards.std() + 1e-8  # Avoid division by zero
    normalized_rewards = (group_rewards - reward_mean) / reward_std
    
    advantages[i:group_end] = normalized_rewards
```

### 2. Clip-Higher (Asymmetric Clipping)

Unlike PPO which uses symmetric clipping, DAPO uses two different clipping thresholds (ε_low < ε_high) to allow the policy to explore more effectively in high-reward regions.

**Implementation**: `models/dapo.py::DAPOAgent.compute_policy_loss`

```python
# Calculate policy gradient losses with asymmetric clipping
pg_losses = []
clip_fracs = []

for i in range(len(self.old_log_probs)):
    ratio = torch.exp(log_probs[i] - self.old_log_probs[i])
    
    if advantages[i] >= 0:
        # For positive advantages, use the higher clipping threshold
        pg_loss = -torch.min(
            ratio * advantages[i],
            torch.clamp(ratio, 1.0 - self.eps_low, 1.0 + self.eps_high) * advantages[i]
        )
    else:
        # For negative advantages, use the lower clipping threshold
        pg_loss = -torch.min(
            ratio * advantages[i],
            torch.clamp(ratio, 1.0 - self.eps_low, 1.0 + self.eps_high) * advantages[i]
        )
    
    pg_losses.append(pg_loss)
    clip_frac = ((ratio - 1.0).abs() > self.eps_low).float().mean().item()
    clip_fracs.append(clip_frac)

pg_loss = torch.mean(torch.stack(pg_losses))
```

### 3. Dynamic Sampling

DADO addresses the "gradient deadzone" problem in RLHF by tracking response variance for each prompt and dynamically skipping prompts that consistently yield 100% or 0% success.

**Implementation**: `models/dapo.py::DAPOAgent.dynamic_sampling`

```python
def dynamic_sampling(self, prompts, prompt_ids, prompt_masks, answer_keys):
    """Dynamically sample prompts based on historical response variance."""
    batch_size = len(prompts)
    selected_indices = []
    skipped_indices = []
    
    for i in range(batch_size):
        prompt = prompts[i]
        variance = self.prompt_variance.get(prompt, None)
        
        # For new prompts or prompts with sufficient variance, include them
        if variance is None or variance > 0.01:  # Minimum variance threshold
            selected_indices.append(i)
        else:
            skipped_indices.append(i)
    
    # If we've skipped more than 50% of prompts, include some randomly
    if len(skipped_indices) > batch_size / 2:
        random_indices = random.sample(
            skipped_indices, 
            k=min(int(batch_size * 0.2), len(skipped_indices))
        )
        selected_indices.extend(random_indices)
    
    # Extract selected items
    selected_prompts = [prompts[i] for i in selected_indices]
    selected_prompt_ids = prompt_ids[selected_indices]
    selected_prompt_masks = prompt_masks[selected_indices]
    selected_answer_keys = [answer_keys[i] for i in selected_indices]
    
    return (
        selected_prompts, 
        selected_prompt_ids, 
        selected_prompt_masks, 
        selected_answer_keys, 
        len(selected_indices), 
        len(skipped_indices)
    )
```

### 4. Token-Level Policy Gradient Loss

DADO computes policy gradient loss across all tokens in all samples, providing more learning signal from long outputs.

**Implementation**: `models/dapo.py::DAPOAgent.compute_policy_loss`

```python
# Compute loss for every individual token in the sequence
for t in range(seq_length):
    if attention_mask[t] == 0:  # Skip padding tokens
        continue
        
    # Get logprobs for this token position
    token_log_probs = log_probs[:, t]
    token_old_log_probs = old_log_probs[:, t]
    
    # Calculate importance ratio
    ratio = torch.exp(token_log_probs - token_old_log_probs)
    
    # Apply clipping based on advantage sign
    if advantages[t] >= 0:
        pg_loss = -torch.min(
            ratio * advantages[t],
            torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages[t]
        )
    else:
        pg_loss = -torch.min(
            ratio * advantages[t],
            torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages[t]
        )
    
    token_losses.append(pg_loss)
```

### 5. Overlong Reward Shaping

DADO includes a soft length penalty to discourage the model from generating excessively long outputs, while still ensuring that good long responses can receive positive rewards.

**Implementation**: `models/reward_model.py::RewardModel.compute_rewards`

```python
# Apply length penalty for excessively long responses
if resp_length > self.target_length * 1.5:  # Allow up to 50% longer
    length_penalty = min(1.0, (resp_length - self.target_length * 1.5) / (self.target_length * 0.5))
    reward = reward * (1.0 - length_penalty * 0.5)  # Reduce reward by up to 50%
    
# For truncated responses, apply a small penalty
if is_truncated:
    truncation_penalty = 0.2
    reward = reward * (1.0 - truncation_penalty)
```

## System Design

### Core Components

1. **PolicyModel (`models/policy_model.py`)**: Wrapper around pretrained language models, handles response generation and probability calculations.

2. **RewardModel (`models/reward_model.py`)**: Computes rewards for generated responses. Supports both neural reward models and rule-based rewards.

3. **DAPOAgent (`models/dapo.py`)**: Implements the DAPO algorithm, coordinating the policy updates, advantage computation, and dynamic sampling.

4. **PromptDataset (`utils/data.py`)**: Handles dataset loading and tokenization for training.

5. **Database Integration (`utils/database.py`)**: Provides SQLite storage for training data, agent interactions, and metrics.

### Training Flow

1. Initialize policy and reward models
2. For each epoch:
   - For each batch of prompts:
     - Apply dynamic sampling to select prompts
     - Generate multiple responses per prompt
     - Compute rewards and advantages
     - Update policy using clipped objective
     - Update prompt variance tracking
   - Save checkpoint

## Optimization Techniques

1. **Batch Processing**: Training examples are processed in batches to maximize GPU utilization.

2. **Memory Efficiency**: Response sequences are generated and processed one at a time to avoid OOM errors with large models.

3. **Gradient Accumulation**: For very large models, gradients can be accumulated across multiple batches before updating.

4. **Checkpointing**: Regular model checkpoints ensure training progress is preserved.

## Extensibility

The implementation is designed to be easily extended:

1. **Custom Reward Models**: Simply create a new reward function or model by extending the RewardModel class.

2. **Alternative Policy Models**: The PolicyModel can be replaced with alternatives by maintaining the same interface.

3. **Configuration**: Most parameters are configurable through the YAML configuration file.

## Best Practices

1. **Start with a Small Group Size**: Begin with a group size of 2-4 and increase as needed.

2. **Tune Clipping Parameters**: Experiment with different values for eps_low and eps_high based on your task.

3. **Monitor Variance Tracking**: Ensure that prompt variance tracking is working correctly and not skipping too many prompts.

4. **Balance Reward Functions**: If using multiple reward components, ensure they are properly balanced and normalized.

5. **Evaluate Frequently**: Regular evaluation on a held-out set helps track progress and detect issues early.
