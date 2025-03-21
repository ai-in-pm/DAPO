import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from tqdm import tqdm

from models.policy_model import PolicyModel
from models.reward_model import RewardModel

class DAPOConfig:
    """Configuration for DAPO algorithm."""
    
    def __init__(self, group_size: int = 4,
                 eps_low: float = 0.2, eps_high: float = 0.3,
                 max_length: int = 512, length_cache: int = 50,
                 learning_rate: float = 1e-5, max_grad_norm: float = 1.0):
        """Initialize DAPO configuration.
        
        Args:
            group_size: Number of outputs to sample per prompt (G).
            eps_low: Lower clipping threshold for PPO.
            eps_high: Upper clipping threshold for PPO.
            max_length: Maximum allowed generation length.
            length_cache: Interval for soft penalty before max_length.
            learning_rate: Learning rate for policy optimization.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        self.group_size = group_size
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.max_length = max_length
        self.length_cache = length_cache
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

class DAPOAgent:
    """Agent implementing the DAPO algorithm."""
    
    def __init__(self, policy_model: PolicyModel, reward_model: RewardModel, config: DAPOConfig):
        """Initialize the DAPO agent.
        
        Args:
            policy_model: Policy model for generating outputs.
            reward_model: Reward model for evaluating outputs.
            config: Configuration for the DAPO algorithm.
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.config = config
        self.device = next(policy_model.parameters()).device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Setup logger
        self.logger = logging.getLogger('dapo')
    
    def training_step(self, prompts: List[str], prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
                    answer_keys: Optional[List[Any]] = None) -> Dict[str, float]:
        """Perform one DAPO training step on a batch of prompts.
        
        Args:
            prompts: List of prompts (questions/tasks) to train on.
            prompt_ids: Tensor of prompt token IDs.
            prompt_mask: Tensor of prompt attention mask.
            answer_keys: Optional list of answer keys for prompts.
            
        Returns:
            Dictionary of training metrics.
        """
        config = self.config
        G = config.group_size
        
        # Move inputs to device
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        
        # 1. Sample G responses for each prompt using the current policy
        old_policy = PolicyModel(self.policy.model.config.name_or_path, 
                                 self.policy.tokenizer.name_or_path)
        old_policy.load_state_dict(self.policy.state_dict())
        old_policy.to(self.device)
        old_policy.eval()
        
        # Generate responses for each prompt
        all_response_ids = []
        all_response_texts = []
        
        self.policy.eval()
        with torch.no_grad():
            for i in range(len(prompts)):
                # Sample G responses for this prompt
                response_ids, response_texts = self.policy.generate(
                    prompt_ids[i:i+1],
                    prompt_mask[i:i+1],
                    max_length=config.max_length,
                    num_return_sequences=G,
                    do_sample=True
                )
                all_response_ids.append(response_ids)
                all_response_texts.append(response_texts)
        
        # 2. Compute rewards for each output
        rewards, truncated = self.reward_model.compute_rewards(
            prompts, all_response_texts, answer_keys
        )
        
        # 3. Dynamic Sampling: filter out prompts with all rewards 1 or all -1
        batch_samples = []  # filtered (prompt_idx, outputs, rewards) tuples
        
        for i, (prompt_rewards, prompt_truncated, response_texts) in enumerate(zip(rewards, truncated, all_response_texts)):
            # Filter out truncated responses
            valid_indices = [j for j, trunc in enumerate(prompt_truncated) if not trunc]
            if not valid_indices:
                # Skip if all responses were truncated
                continue
                
            # Consider only non-truncated outputs for dynamic sampling
            valid_rewards = [prompt_rewards[j] for j in valid_indices]
            valid_responses = [response_texts[j] for j in valid_indices]
            
            # Dynamic sampling: skip if all rewards are same sign (no learning signal)
            if all(r > 0 for r in valid_rewards) or all(r <= 0 for r in valid_rewards):
                continue
                
            # If we reach here, this prompt has a mix of rewards -> valid for training
            batch_samples.append((i, valid_responses, valid_rewards))
            
        # Check if we have any valid samples
        if not batch_samples:
            self.logger.warning("No prompts with variance in rewards - skipping update")
            return {"loss": 0.0, "n_prompts": 0, "mean_reward": 0.0, "reward_std": 0.0}
        
        # 4. Compute advantages and perform policy update
        self.policy.train()
        old_policy.eval()
        
        # Prepare for policy optimization
        total_loss = 0.0
        total_tokens = 0
        total_reward = 0.0
        rewards_list = []
        
        # Process each valid prompt separately
        for prompt_idx, responses, response_rewards in batch_samples:
            # Calculate mean and std for group advantage normalization
            mean_reward = np.mean(response_rewards)
            std_reward = np.std(response_rewards) + 1e-8  # Small epsilon to avoid division by zero
            
            # Process each response in the group
            for response_text, reward in zip(responses, response_rewards):
                # Calculate advantage for this response
                advantage = (reward - mean_reward) / std_reward
                
                # Tokenize the prompt+response
                full_text = prompts[prompt_idx] + response_text
                inputs = self.policy.tokenizer(
                    full_text,
                    return_tensors='pt',
                    max_length=config.max_length,
                    padding='max_length',
                    truncation=True
                ).to(self.device)
                
                # Get input ids and mask
                input_ids = inputs['input_ids']
                attn_mask = inputs['attention_mask']
                
                # Compute token-level log probs for old and new policies
                with torch.no_grad():
                    old_log_probs = old_policy.get_token_logprobs(input_ids, attn_mask)
                
                new_log_probs = self.policy.get_token_logprobs(input_ids, attn_mask)
                
                # Find the prompt length to exclude prompt tokens from policy update
                prompt_length = len(self.policy.tokenizer.encode(prompts[prompt_idx]))
                
                # Calculate ratios for token-level PPO
                # Only consider response tokens (exclude prompt tokens)
                old_log_probs_response = old_log_probs[:, prompt_length-1:]
                new_log_probs_response = new_log_probs[:, prompt_length-1:]
                response_mask = attn_mask[:, prompt_length:]
                
                # Calculate probability ratios
                ratio = torch.exp(new_log_probs_response - old_log_probs_response)
                
                # Apply Clip-Higher: asymmetric clipping
                if advantage >= 0:
                    # Positive advantage: clip ratio above with eps_high
                    ratio_clipped = torch.min(ratio, torch.tensor(1.0 + config.eps_high, device=self.device))
                else:
                    # Negative advantage: clip ratio below with eps_low
                    ratio_clipped = torch.max(ratio, torch.tensor(1.0 - config.eps_low, device=self.device))
                
                # Compute token-level losses (scaled by advantage)
                token_loss = -torch.min(
                    ratio * advantage,
                    ratio_clipped * advantage
                )
                
                # Apply response mask and sum token-level losses
                masked_token_loss = token_loss * response_mask
                response_loss = masked_token_loss.sum()
                
                # Count tokens for loss normalization
                n_tokens = response_mask.sum().item()
                
                # Accumulate loss
                total_loss += response_loss
                total_tokens += n_tokens
                total_reward += reward
                rewards_list.append(reward)
        
        # Normalize loss by token count and perform gradient update
        if total_tokens > 0:
            loss = total_loss / total_tokens
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "n_prompts": len(batch_samples),
            "mean_reward": float(total_reward) / max(1, len(rewards_list)),
            "reward_std": float(np.std(rewards_list)) if rewards_list else 0.0
        }
        
        return metrics
    
    def save(self, save_dir: str) -> None:
        """Save the DAPO agent.
        
        Args:
            save_dir: Directory to save to.
        """
        self.policy.save(save_dir)
        
    @classmethod
    def load(cls, load_dir: str, reward_model: RewardModel, config: DAPOConfig) -> 'DAPOAgent':
        """Load a DAPO agent from a directory.
        
        Args:
            load_dir: Directory to load from.
            reward_model: Reward model to use.
            config: Configuration for the DAPO algorithm.
            
        Returns:
            Loaded DAPOAgent instance.
        """
        policy_model = PolicyModel.load(load_dir)
        return cls(policy_model, reward_model, config)
