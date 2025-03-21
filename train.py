import os
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
import yaml
from typing import Dict, Any

from models.policy_model import PolicyModel
from models.reward_model import RewardModel
from models.dapo import DAPOAgent, DAPOConfig
from utils.config import Config
from utils.logging import setup_logger
from utils.data import PromptDataset

def train(config: Config) -> None:
    """Train a policy model using DAPO.
    
    Args:
        config: Training configuration.
    """
    # Setup logger
    log_dir = config.get('system.log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('dapo_train', log_dir)
    
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
    model_name = config.get('model.name', 'gpt2')
    tokenizer_name = config.get('model.tokenizer', 'gpt2')
    logger.info(f"Loading policy model: {model_name}")
    policy_model = PolicyModel(model_name, tokenizer_name)
    policy_model.to(device)
    
    # Load reward model (or use rule-based rewards)
    reward_model_name = config.get('model.reward_model', None)
    reward_tokenizer_name = config.get('model.reward_tokenizer', None)
    max_length = config.get('learning.max_length', 512)
    
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
    
    # Create DAPO config
    dapo_config = DAPOConfig(
        group_size=config.get('learning.group_size', 4),
        eps_low=config.get('learning.eps_low', 0.2),
        eps_high=config.get('learning.eps_high', 0.3),
        max_length=max_length,
        length_cache=config.get('learning.length_cache', 50),
        learning_rate=config.get('learning.learning_rate', 1e-5),
        max_grad_norm=config.get('learning.max_grad_norm', 1.0)
    )
    
    # Create DAPO agent
    dapo_agent = DAPOAgent(policy_model, reward_model, dapo_config)
    
    # Load dataset
    train_file = config.get('data.train_file', 'data/train.jsonl')
    logger.info(f"Loading dataset from {train_file}")
    tokenizer = policy_model.tokenizer
    
    dataloader = PromptDataset.create_dataloader(
        train_file,
        tokenizer,
        batch_size=config.get('learning.batch_size', 16),
        shuffle=True,
        max_length=max_length
    )
    
    # Setup checkpoint directory
    checkpoint_dir = config.get('system.checkpoint_dir', 'models/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    num_epochs = config.get('learning.num_epochs', 10)
    log_interval = config.get('logging.log_interval', 10)
    save_interval = config.get('logging.save_interval', 500)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Extract batch data
            prompts = batch['prompts']
            prompt_ids = batch['prompt_ids']
            prompt_mask = batch['prompt_mask']
            answer_keys = batch['answer_keys']
            
            # Perform one DAPO training step
            metrics = dapo_agent.training_step(prompts, prompt_ids, prompt_mask, answer_keys)
            
            # Update epoch metrics
            epoch_loss += metrics['loss']
            epoch_reward += metrics['mean_reward']
            num_batches += 1
            global_step += 1
            
            # Log metrics
            if global_step % log_interval == 0:
                logger.info(
                    f"Step {global_step}: loss={metrics['loss']:.4f}, "
                    f"reward={metrics['mean_reward']:.4f}, "
                    f"reward_std={metrics['reward_std']:.4f}, "
                    f"n_prompts={metrics['n_prompts']}"
                )
            
            # Save checkpoint
            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"dapo_step_{global_step}")
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                dapo_agent.save(checkpoint_path)
        
        # Compute epoch metrics
        epoch_loss /= max(1, num_batches)
        epoch_reward /= max(1, num_batches)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} completed: "
            f"loss={epoch_loss:.4f}, reward={epoch_reward:.4f}"
        )
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"dapo_epoch_{epoch+1}")
        logger.info(f"Saving epoch checkpoint to {checkpoint_path}")
        dapo_agent.save(checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "dapo_final")
    logger.info(f"Training completed. Saving final model to {final_model_path}")
    dapo_agent.save(final_model_path)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a policy model using DAPO")
    parser.add_argument(
        '--config', type=str, default='config/default.yaml',
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Start training
    train(config)

if __name__ == '__main__':
    main()
