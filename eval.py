import os
import argparse
import logging
import torch
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, List, Any, Optional

from models.policy_model import PolicyModel
from models.reward_model import RewardModel
from models.dapo import DAPOAgent, DAPOConfig
from utils.config import Config
from utils.logging import setup_logger
from utils.data import PromptDataset

def evaluate(model_path: str, eval_file: str, output_file: Optional[str] = None,
           config_path: str = 'config/default.yaml') -> Dict[str, float]:
    """Evaluate a trained policy model.
    
    Args:
        model_path: Path to the trained model directory.
        eval_file: Path to the evaluation dataset file.
        output_file: Optional path to save evaluation results.
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Load configuration
    config = Config(config_path)
    
    # Setup logger
    log_dir = config.get('system.log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('dapo_eval', log_dir)
    
    # Set random seed for reproducibility
    seed = config.get('system.seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device_name = config.get('system.device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    # Load policy model
    logger.info(f"Loading policy model from {model_path}")
    policy_model = PolicyModel.load(model_path)
    policy_model.to(device)
    policy_model.eval()
    
    # Configuration for evaluation
    max_length = config.get('learning.max_length', 512)
    
    # Create reward model for evaluation
    reward_model_name = config.get('model.reward_model', None)
    reward_tokenizer_name = config.get('model.reward_tokenizer', None)
    
    if reward_model_name:
        logger.info(f"Loading reward model: {reward_model_name}")
        reward_model = RewardModel(
            reward_model_name, 
            reward_tokenizer_name, 
            max_length, 
            device_name
        )
    else:
        logger.info("Using rule-based reward model")
        reward_model = RewardModel(
            model_name=None,
            tokenizer_name=None,
            max_length=max_length,
            device=device_name
        )
    
    # Load evaluation dataset
    logger.info(f"Loading evaluation dataset from {eval_file}")
    eval_dataset = PromptDataset(eval_file, policy_model.tokenizer, max_length)
    
    # Evaluation metrics
    results = {
        'total_samples': len(eval_dataset),
        'correct_count': 0,
        'total_reward': 0.0,
        'rewards': [],
        'prompts': [],
        'responses': [],
        'rewards_by_prompt': []
    }
    
    # Evaluate each prompt
    logger.info("Starting evaluation")
    
    for i in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        item = eval_dataset[i]
        prompt = item['prompt']
        prompt_ids = item['prompt_ids'].unsqueeze(0).to(device)
        prompt_mask = item['prompt_mask'].unsqueeze(0).to(device)
        answer_key = item['answer_key']
        
        # Generate response with the policy model
        with torch.no_grad():
            _, response_texts = policy_model.generate(
                prompt_ids,
                prompt_mask,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,  # Lower temperature for evaluation
                do_sample=True
            )
        
        response = response_texts[0] if response_texts else ""
        
        # Calculate reward for the response
        rewards, truncated = reward_model.compute_rewards([prompt], [[response]], [answer_key])
        reward = rewards[0][0] if rewards and rewards[0] else 0.0
        truncated = truncated[0][0] if truncated and truncated[0] else False
        
        # Update metrics
        results['total_reward'] += reward
        results['rewards'].append(reward)
        results['prompts'].append(prompt)
        results['responses'].append(response)
        results['rewards_by_prompt'].append({
            'prompt': prompt,
            'response': response,
            'reward': reward,
            'truncated': truncated,
            'answer_key': answer_key
        })
        
        # Determine correctness (reward > 0 as a simple heuristic)
        if reward > 0:
            results['correct_count'] += 1
    
    # Compute aggregated metrics
    results['accuracy'] = results['correct_count'] / max(1, results['total_samples'])
    results['mean_reward'] = results['total_reward'] / max(1, results['total_samples'])
    results['reward_std'] = float(np.std(results['rewards'])) if results['rewards'] else 0.0
    
    # Log results
    logger.info(f"Evaluation completed:")
    logger.info(f"  Samples: {results['total_samples']}")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Mean Reward: {results['mean_reward']:.4f}")
    logger.info(f"  Reward Std: {results['reward_std']:.4f}")
    
    # Save results to file if specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {output_file}")
    
    return {
        'accuracy': results['accuracy'],
        'mean_reward': results['mean_reward'],
        'reward_std': results['reward_std']
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained policy model")
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--eval-file', type=str, default='data/eval.jsonl',
        help='Path to the evaluation dataset file'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--config', type=str, default='config/default.yaml',
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(args.model, args.eval_file, args.output, args.config)

if __name__ == '__main__':
    main()
