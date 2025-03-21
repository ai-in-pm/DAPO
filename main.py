import os
import argparse
import logging
import torch
import numpy as np
from typing import List, Optional

from models.policy_model import PolicyModel
from models.reward_model import RewardModel
from models.dapo import DAPOConfig
from utils.config import Config
from utils.logging import setup_logger
from utils.database import DAPODatabase

def generate_response(prompt: str, model_path: str, config_path: str = 'config/default.yaml',
                    temperature: float = 0.7, num_samples: int = 1,
                    max_length: Optional[int] = None) -> List[str]:
    """Generate responses for a prompt using a trained model.
    
    Args:
        prompt: Input prompt text.
        model_path: Path to the trained model directory.
        config_path: Path to the configuration file.
        temperature: Sampling temperature.
        num_samples: Number of responses to generate.
        max_length: Maximum response length (or None to use config value).
        
    Returns:
        List of generated response texts.
    """
    # Load configuration
    config = Config(config_path)
    
    # Setup logger
    log_dir = config.get('system.log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('dapo_inference', log_dir)
    
    # Setup device
    device_name = config.get('system.device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    # Load policy model
    logger.info(f"Loading policy model from {model_path}")
    policy_model = PolicyModel.load(model_path)
    policy_model.to(device)
    policy_model.eval()
    
    # Configuration for generation
    if max_length is None:
        max_length = config.get('learning.max_length', 512)
    
    # Prepare input tokens
    inputs = policy_model.tokenizer(
        prompt,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    ).to(device)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Generate responses
    logger.info(f"Generating {num_samples} response(s) for prompt: {prompt}")
    
    with torch.no_grad():
        _, response_texts = policy_model.generate(
            input_ids,
            attention_mask,
            max_length=max_length,
            num_return_sequences=num_samples,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )
    
    return response_texts

def run_interactive_mode(model_path: str, config_path: str = 'config/default.yaml'):
    """Run interactive mode for testing the model.
    
    Args:
        model_path: Path to the trained model directory.
        config_path: Path to the configuration file.
    """
    # Load configuration
    config = Config(config_path)
    
    # Setup logger
    log_dir = config.get('system.log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('dapo_interactive', log_dir)
    
    # Load policy model
    logger.info(f"Loading policy model from {model_path}")
    policy_model = PolicyModel.load(model_path)
    device_name = config.get('system.device', 'cuda' if torch.cuda.is_available() else 'cpu')
    policy_model.to(torch.device(device_name))
    policy_model.eval()
    
    # Configuration for generation
    max_length = config.get('learning.max_length', 512)
    
    # Create database connection
    db_path = config.get('system.database', 'data/dapo.db')
    database = DAPODatabase(db_path)
    
    # Display welcome message
    print("\n" + "=" * 80)
    print(f"Interactive DAPO Agent - Model: {model_path}")
    print("Enter a prompt to generate a response, or 'quit' to exit.")
    print("=" * 80 + "\n")
    
    # Interactive loop
    while True:
        try:
            prompt = input("\nPrompt> ")
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
                
            # Generate response
            responses = generate_response(
                prompt, 
                model_path, 
                config_path,
                temperature=0.7, 
                num_samples=1,
                max_length=max_length
            )
            
            # Display response
            print("\nResponse:")
            response = responses[0] if responses else "[No response generated]"                
            print(f"{response}")
            
            # Save interaction to database
            database.save_interaction(prompt, response)
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nInteractive session ended. Goodbye!")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DAPO AI Agent")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate a response')
    gen_parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    gen_parser.add_argument('--model', type=str, required=True, help='Path to the model directory')
    gen_parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    gen_parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    gen_parser.add_argument('--num-samples', type=int, default=1, help='Number of responses to generate')
    gen_parser.add_argument('--max-length', type=int, default=None, help='Maximum response length')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run in interactive mode')
    interactive_parser.add_argument('--model', type=str, required=True, help='Path to the model directory')
    interactive_parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # Generate a response
        responses = generate_response(
            args.prompt,
            args.model,
            args.config,
            args.temperature,
            args.num_samples,
            args.max_length
        )
        
        print("\nGenerated responses:")
        for i, response in enumerate(responses):
            print(f"\nResponse {i+1}:\n{response}")
            
    elif args.command == 'interactive':
        # Run interactive mode
        run_interactive_mode(args.model, args.config)
        
    else:
        # No command specified, display help
        parser.print_help()

if __name__ == '__main__':
    main()
