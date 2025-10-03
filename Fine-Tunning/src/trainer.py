"""
Main training script for fine-tuning project.
"""
import os
import torch
import wandb
from typing import Optional, Dict, Any
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt

from .config import Config
from .data_utils import DataManager
from .model_utils import ModelManager
from .evaluation import Evaluator


class FineTuningTrainer:
    """Main training class for fine-tuning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer."""
        self.config = Config(config_path)
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.evaluator = None
        self.trainer = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging (wandb, tensorboard)."""
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.project_name,
                config=self.config.to_dict()
            )
    
    def prepare_data(self, data_source: str = "sample", data_path: Optional[str] = None) -> tuple:
        """Prepare data for training."""
        print("📊 Preparing data...")
        
        # Load data
        if data_source == "sample":
            data = self.data_manager.create_sample_data(num_samples=1000)
        elif data_source == "json" and data_path:
            data = self.data_manager.load_data_from_json(data_path)
        elif data_source == "csv" and data_path:
            data = self.data_manager.load_data_from_csv(data_path)
        else:
            raise ValueError("Invalid data source or missing data path")
        
        # Print data statistics
        stats = self.data_manager.get_data_stats(data)
        print(f"Data Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.data_manager.create_datasets(data)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.data_manager.create_dataloaders()
        
        print(f"✅ Data prepared:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    
    def setup_model(self, token: str) -> tuple:
        """Setup model and tokenizer."""
        print("🤖 Setting up model...")
        
        # Load model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer(
            self.config.model.name, 
            token
        )
        
        # Setup LoRA
        peft_model = self.model_manager.setup_lora()
        
        # Initialize evaluator
        self.evaluator = Evaluator(tokenizer)
        
        print("✅ Model setup complete")
        return model, tokenizer, peft_model
    
    def train(self, train_dataset, val_dataset, output_dir: str = None) -> Trainer:
        """Train the model."""
        print("🚀 Starting training...")
        
        if output_dir is None:
            output_dir = os.path.join(self.config.paths.model_save_path, "training_output")
        
        # Setup training arguments
        training_args = self.model_manager.setup_training_args(output_dir)
        
        # Create trainer
        self.trainer = self.model_manager.create_trainer(
            train_dataset, 
            val_dataset, 
            training_args
        )
        
        # Train
        self.trainer.train()
        
        print("✅ Training complete")
        return self.trainer
    
    def evaluate(self, test_dataset, test_loader) -> Dict[str, Any]:
        """Evaluate the model."""
        print("📈 Evaluating model...")
        
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate perplexity
        perplexity = self.evaluator.calculate_perplexity(
            self.trainer.model, 
            test_loader
        )
        
        # Evaluate generation quality
        test_data = []
        for i in range(min(50, len(test_dataset))):
            item = test_dataset.data[i]
            test_data.append(item)
        
        generation_results = self.evaluator.evaluate_generation_quality(
            self.trainer.model,
            test_data
        )
        
        # Combine results
        results = {
            'perplexity': perplexity,
            'bleu': generation_results['bleu'],
            'rouge': generation_results['rouge'],
            'predictions': generation_results['predictions'],
            'references': generation_results['references']
        }
        
        print(f"📊 Evaluation Results:")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  BLEU: {results['bleu']:.4f}")
        print(f"  ROUGE-1: {results['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['rouge']['rougeL']:.4f}")
        
        return results
    
    def save_model(self, save_path: str = None):
        """Save the trained model."""
        if save_path is None:
            save_path = os.path.join(self.config.paths.model_save_path, "finetuned_model")
        
        print(f"💾 Saving model to {save_path}...")
        self.model_manager.save_model(save_path)
        print("✅ Model saved")
    
    def generate_examples(self, prompts: list, max_new_tokens: int = 100) -> list:
        """Generate examples for given prompts."""
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("🎯 Generating examples...")
        
        responses = []
        for prompt in prompts:
            response = self.model_manager.generate_response(
                prompt, 
                max_new_tokens=max_new_tokens
            )
            responses.append(response)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 50)
        
        return responses
    
    def create_training_report(self, results: Dict[str, Any], save_path: str = None):
        """Create comprehensive training report."""
        if save_path is None:
            save_path = os.path.join(self.config.paths.output_path, "training_report.md")
        
        print("📝 Creating training report...")
        
        # Create evaluation report
        eval_report = self.evaluator.create_evaluation_report(results)
        
        # Add training information
        model_info = self.model_manager.get_model_info()
        
        full_report = f"""
# Fine-tuning Training Report

## Model Information
- **Model**: {model_info['model_name']}
- **Total Parameters**: {model_info['total_parameters']:,}
- **Trainable Parameters**: {model_info['trainable_parameters']:,}
- **Trainable Percentage**: {model_info['trainable_percentage']:.2f}%
- **Device**: {model_info['device']}
- **Data Type**: {model_info['dtype']}

## Training Configuration
- **Learning Rate**: {self.config.training.learning_rate}
- **Batch Size**: {self.config.training.batch_size}
- **Epochs**: {self.config.training.num_epochs}
- **LoRA Rank**: {self.config.lora.r}
- **LoRA Alpha**: {self.config.lora.lora_alpha}

{eval_report}
"""
        
        with open(save_path, 'w') as f:
            f.write(full_report)
        
        print(f"✅ Training report saved to {save_path}")
        return full_report
    
    def run_full_pipeline(self, token: str, data_source: str = "sample", data_path: Optional[str] = None):
        """Run the complete fine-tuning pipeline."""
        print("🎯 Starting full fine-tuning pipeline...")
        
        try:
            # Prepare data
            train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = self.prepare_data(
                data_source, data_path
            )
            
            # Setup model
            model, tokenizer, peft_model = self.setup_model(token)
            
            # Train
            trainer = self.train(train_dataset, val_dataset)
            
            # Evaluate
            results = self.evaluate(test_dataset, test_loader)
            
            # Save model
            self.save_model()
            
            # Create report
            self.create_training_report(results)
            
            # Generate some examples
            example_prompts = [
                "What is machine learning?",
                "Explain neural networks",
                "How does deep learning work?"
            ]
            self.generate_examples(example_prompts)
            
            print("🎉 Full pipeline completed successfully!")
            
            return {
                'trainer': trainer,
                'results': results,
                'model_info': self.model_manager.get_model_info()
            }
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise
        finally:
            # Cleanup
            if self.config.logging.use_wandb:
                wandb.finish()
