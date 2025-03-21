import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel:
    """Reward model for DAPO algorithm."""
    
    def __init__(self, model_name: Optional[str] = None, tokenizer_name: Optional[str] = None, 
                 max_length: int = 512, device: str = 'cuda'):
        """Initialize the reward model.
        
        Can be initialized either with a pretrained model or as a rule-based model.
        
        Args:
            model_name: Name of the pretrained model. If None, uses rule-based rewards.
            tokenizer_name: Name of the tokenizer. If None, uses rule-based rewards.
            max_length: Maximum sequence length.
            device: Device to use.
        """
        self.device = device
        self.max_length = max_length
        self.use_neural_model = model_name is not None and tokenizer_name is not None
        
        if self.use_neural_model:
            # Neural reward model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Rule-based reward model
            self.model = None
            self.tokenizer = None
            
    def compute_rewards(self, prompts: List[str], responses: List[List[str]], 
                      answer_keys: Optional[List[Any]] = None) -> Tuple[List[List[float]], List[List[bool]]]:
        """Compute rewards for response groups.
        
        Args:
            prompts: List of prompts.
            responses: List of response groups, where each group is a list of responses for a prompt.
            answer_keys: Optional list of answer keys for rule-based rewards.
            
        Returns:
            Tuple of:
                - List of reward lists for each prompt's responses
                - List of truncation flags for each prompt's responses
        """
        all_rewards = []
        all_truncated = []
        
        for i, (prompt, response_group) in enumerate(zip(prompts, responses)):
            group_rewards = []
            group_truncated = []
            
            for response in response_group:
                if self.use_neural_model:
                    # Neural reward computation
                    reward, truncated = self._compute_neural_reward(prompt, response)
                else:
                    # Rule-based reward computation
                    answer_key = answer_keys[i] if answer_keys else None
                    reward, truncated = self._compute_rule_based_reward(prompt, response, answer_key)
                    
                group_rewards.append(reward)
                group_truncated.append(truncated)
                
            all_rewards.append(group_rewards)
            all_truncated.append(group_truncated)
            
        return all_rewards, all_truncated
    
    def _compute_neural_reward(self, prompt: str, response: str) -> Tuple[float, bool]:
        """Compute reward using neural reward model.
        
        Args:
            prompt: Prompt text.
            response: Response text.
            
        Returns:
            Tuple of reward value and truncation flag.
        """
        # Check if response is too long
        tokens = self.tokenizer.encode(response)
        truncated = len(tokens) > self.max_length
        
        # Prepare input for the reward model
        inputs = self.tokenizer(
            prompt, response,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        ).to(self.device)
        
        # Get reward prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward_value = outputs.logits.item()
            
        # Apply overlong punishment (length-based penalty)
        length_penalty = self._compute_length_penalty(tokens)
        final_reward = reward_value + length_penalty
        
        return final_reward, truncated
    
    def _compute_rule_based_reward(self, prompt: str, response: str, 
                                answer_key: Optional[Any] = None) -> Tuple[float, bool]:
        """Compute rule-based reward.
        
        Args:
            prompt: Prompt text.
            response: Response text.
            answer_key: Optional answer key for correctness evaluation.
            
        Returns:
            Tuple of reward value and truncation flag.
        """
        # Simple token count for length check
        tokens = response.split()
        truncated = len(tokens) > self.max_length
        
        # Base reward (could be enhanced with more sophisticated rules)
        if answer_key is not None:
            # If we have an answer key, check correctness
            if isinstance(answer_key, str) and answer_key.strip().lower() in response.strip().lower():
                base_reward = 1.0
            elif isinstance(answer_key, (list, tuple)) and any(ans.strip().lower() in response.strip().lower() for ans in answer_key):
                base_reward = 1.0
            else:
                base_reward = -1.0
        else:
            # Quality-based reward when no answer key is available
            # Implement basic quality heuristics
            # This is simplified - real implementation would use more sophisticated measures
            words = response.split()
            if len(words) < 5:  # Too short
                base_reward = -0.5
            elif len(set(words)) / len(words) < 0.4:  # Too repetitive
                base_reward = -0.7
            elif len(words) > 100:  # Verbose but not truncated
                base_reward = 0.0
            else:  # Default moderate reward
                base_reward = 0.3
        
        # Apply overlong punishment (length-based penalty)
        length_penalty = self._compute_length_penalty(tokens)
        final_reward = base_reward + length_penalty
        
        return final_reward, truncated
    
    def _compute_length_penalty(self, tokens: List[Any]) -> float:
        """Compute length-based penalty for overlong responses.
        
        Args:
            tokens: List of tokens (or token IDs).
            
        Returns:
            Length penalty value (negative or zero).
        """
        token_count = len(tokens)
        
        # Define thresholds for length penalty
        max_length = self.max_length
        length_cache = max(10, int(0.2 * max_length))  # Default to 20% of max_length
        
        # Compute penalty
        if token_count > max_length:
            # Maximum penalty for exceeding max_length
            return -1.0
        elif token_count > max_length - length_cache:
            # Linear penalty within the cache interval
            over_length = token_count - (max_length - length_cache)
            return -over_length / length_cache
        else:
            # No penalty for normal-length responses
            return 0.0
