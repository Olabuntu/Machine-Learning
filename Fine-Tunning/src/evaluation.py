"""
Evaluation utilities for fine-tuning project.
"""
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer
import nltk
from rouge_score import rouge_scorer
import sacrebleu
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """Evaluation class for fine-tuned models."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        """Initialize evaluator."""
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def calculate_perplexity(self, model, dataloader) -> float:
        """Calculate perplexity on a dataset."""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].numel()
                total_tokens += batch['input_ids'].numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score."""
        # Tokenize predictions and references
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        # Calculate BLEU
        bleu_score = sacrebleu.corpus_bleu(predictions, ref_tokens)
        return bleu_score.score
    
    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Calculate average scores
        avg_scores = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        return avg_scores
    
    def evaluate_generation_quality(self, model, test_data: List[Dict], max_samples: int = 50) -> Dict[str, Any]:
        """Evaluate generation quality on test data."""
        model.eval()
        
        # Sample data if too large
        if len(test_data) > max_samples:
            test_data = test_data[:max_samples]
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for item in test_data:
                # Generate response
                prompt = item['prompt']
                messages = [{"role": "user", "content": prompt}]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = response[len(text):].strip()
                
                predictions.append(generated_text)
                references.append(item['response'])
        
        # Calculate metrics
        bleu_score = self.calculate_bleu(predictions, references)
        rouge_scores = self.calculate_rouge(predictions, references)
        
        return {
            'bleu': bleu_score,
            'rouge': rouge_scores,
            'predictions': predictions,
            'references': references
        }
    
    def create_evaluation_report(self, results: Dict[str, Any], save_path: str = None):
        """Create a comprehensive evaluation report."""
        report = f"""
# Model Evaluation Report

## Metrics Summary
- **BLEU Score**: {results.get('bleu', 'N/A'):.4f}
- **ROUGE-1 F1**: {results.get('rouge', {}).get('rouge1', 'N/A'):.4f}
- **ROUGE-2 F1**: {results.get('rouge', {}).get('rouge2', 'N/A'):.4f}
- **ROUGE-L F1**: {results.get('rouge', {}).get('rougeL', 'N/A'):.4f}

## Sample Predictions
"""
        
        # Add sample predictions
        predictions = results.get('predictions', [])
        references = results.get('references', [])
        
        for i in range(min(5, len(predictions))):
            report += f"""
### Sample {i+1}
**Reference**: {references[i]}
**Prediction**: {predictions[i]}
---
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_training_metrics(self, train_losses: List[float], val_losses: List[float], save_path: str = None):
        """Plot training metrics."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]], save_path: str = None):
        """Compare multiple models."""
        models = list(model_results.keys())
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = []
            for model in models:
                if metric == 'bleu':
                    values.append(model_results[model].get('bleu', 0))
                else:
                    values.append(model_results[model].get('rouge', {}).get(metric, 0))
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.upper()} Score')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_response_lengths(self, predictions: List[str], references: List[str], save_path: str = None):
        """Analyze response length distributions."""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(pred_lengths, bins=20, alpha=0.7, label='Predictions')
        plt.hist(ref_lengths, bins=20, alpha=0.7, label='References')
        plt.xlabel('Response Length (words)')
        plt.ylabel('Frequency')
        plt.title('Response Length Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(ref_lengths, pred_lengths, alpha=0.6)
        plt.xlabel('Reference Length')
        plt.ylabel('Prediction Length')
        plt.title('Length Correlation')
        
        # Add diagonal line
        max_len = max(max(ref_lengths), max(pred_lengths))
        plt.plot([0, max_len], [0, max_len], 'r--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_correlation': np.corrcoef(ref_lengths, pred_lengths)[0, 1]
        }
