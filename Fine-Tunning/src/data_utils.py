"""
Data utilities for fine-tuning project.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import json
import os
from sklearn.model_selection import train_test_split


class FineTuningDataset(Dataset):
    """Custom dataset for fine-tuning."""
    
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            data: List of dictionaries containing 'prompt' and 'response' keys
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = self.data[idx]
        
        # Create chat template
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokenized["input_ids"].clone()
        
        # Mask the prompt part (only train on response)
        prompt_end = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_tokens = self.tokenizer(
            prompt_end,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"]
        
        # Set prompt tokens to -100 (ignore in loss)
        labels[0, :prompt_tokens.shape[1]] = -100
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


class DataManager:
    """Data management class."""
    
    def __init__(self, config):
        """Initialize data manager."""
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side=self.config.model.padding_side
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.tokenizer
    
    def load_data_from_json(self, file_path: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def load_data_from_csv(self, file_path: str, prompt_col: str = "prompt", response_col: str = "response") -> List[Dict]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        data = []
        
        for _, row in df.iterrows():
            data.append({
                "prompt": row[prompt_col],
                "response": row[response_col]
            })
        
        return data
    
    def create_sample_data(self, num_samples: int = 100) -> List[Dict]:
        """Create sample data for testing."""
        sample_data = []
        
        prompts = [
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?",
            "What is natural language processing?",
            "Explain computer vision",
            "What is reinforcement learning?",
            "How do transformers work?",
            "What is fine-tuning?",
            "Explain attention mechanisms",
            "What is transfer learning?"
        ]
        
        responses = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "Natural language processing is a field of AI that focuses on enabling computers to understand, interpret, and generate human language.",
            "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world.",
            "Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment.",
            "Transformers are neural network architectures that use attention mechanisms to process sequential data efficiently.",
            "Fine-tuning is the process of adapting a pre-trained model to a specific task or domain.",
            "Attention mechanisms allow models to focus on relevant parts of the input when making predictions.",
            "Transfer learning is a technique where knowledge gained from one task is applied to a related task."
        ]
        
        for i in range(num_samples):
            sample_data.append({
                "prompt": prompts[i % len(prompts)],
                "response": responses[i % len(responses)]
            })
        
        return sample_data
    
    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets."""
        # First split: train + val vs test
        train_val_data, test_data = train_test_split(
            data, 
            test_size=self.config.data.test_split,
            random_state=42
        )
        
        # Second split: train vs val
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=self.config.data.val_split / (self.config.data.train_split + self.config.data.val_split),
            random_state=42
        )
        
        return train_data, val_data, test_data
    
    def create_datasets(self, data: List[Dict]) -> Tuple[FineTuningDataset, FineTuningDataset, FineTuningDataset]:
        """Create train, validation, and test datasets."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        # Split data
        train_data, val_data, test_data = self.split_data(data)
        
        # Create datasets
        self.train_dataset = FineTuningDataset(
            train_data, 
            self.tokenizer, 
            self.config.model.max_length
        )
        
        self.val_dataset = FineTuningDataset(
            val_data, 
            self.tokenizer, 
            self.config.model.max_length
        )
        
        self.test_dataset = FineTuningDataset(
            test_data, 
            self.tokenizer, 
            self.config.model.max_length
        )
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders."""
        if any(dataset is None for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]):
            raise ValueError("Datasets not created. Call create_datasets() first.")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def save_data(self, data: List[Dict], file_path: str):
        """Save data to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_data_stats(self, data: List[Dict]) -> Dict:
        """Get statistics about the data."""
        prompt_lengths = [len(item["prompt"]) for item in data]
        response_lengths = [len(item["response"]) for item in data]
        
        return {
            "total_samples": len(data),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_response_length": sum(response_lengths) / len(response_lengths),
            "max_prompt_length": max(prompt_lengths),
            "max_response_length": max(response_lengths),
            "min_prompt_length": min(prompt_lengths),
            "min_response_length": min(response_lengths)
        }
