import json
import os
from typing import Dict, List, Tuple, Any, Optional, Iterator
import random
import torch
from torch.utils.data import Dataset, DataLoader

class PromptDataset(Dataset):
    """Dataset for prompt-response pairs used in DAPO training."""
    
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512):
        """Initialize the dataset from a JSONL file.
        
        Args:
            data_path: Path to the JSONL file containing prompt-response pairs.
            tokenizer: Tokenizer for encoding prompts and responses.
            max_length: Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data from JSONL file
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'prompt' in item and 'responses' in item:
                    # Original format with multiple responses per prompt
                    self.data.append({
                        'prompt': item['prompt'],
                        'responses': item['responses'],
                        'answer_key': item.get('answer_key', None)
                    })
                elif 'prompt' in item and 'response' in item:
                    # Alternative format with single response
                    self.data.append({
                        'prompt': item['prompt'],
                        'responses': [item['response']],
                        'answer_key': item.get('answer_key', None)
                    })
                    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Encode the prompt for the model
        prompt_encoding = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'prompt': item['prompt'],
            'prompt_ids': prompt_encoding['input_ids'].squeeze(0),
            'prompt_mask': prompt_encoding['attention_mask'].squeeze(0),
            'responses': item['responses'],
            'answer_key': item['answer_key']
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for DataLoader.
        
        Args:
            batch: Batch of dataset items.
            
        Returns:
            Collated batch dictionary.
        """
        return {
            'prompts': [item['prompt'] for item in batch],
            'prompt_ids': torch.stack([item['prompt_ids'] for item in batch]),
            'prompt_mask': torch.stack([item['prompt_mask'] for item in batch]),
            'responses': [item['responses'] for item in batch],
            'answer_keys': [item['answer_key'] for item in batch]
        }
    
    @classmethod
    def create_dataloader(
        cls, data_path: str, tokenizer: Any, batch_size: int = 8, 
        shuffle: bool = True, max_length: int = 512
    ) -> DataLoader:
        """Create a DataLoader for the dataset.
        
        Args:
            data_path: Path to the JSONL file containing prompt-response pairs.
            tokenizer: Tokenizer for encoding prompts and responses.
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.
            max_length: Maximum sequence length.
            
        Returns:
            DataLoader for the dataset.
        """
        dataset = cls(data_path, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=cls.collate_fn
        )
