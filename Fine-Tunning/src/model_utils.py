"""
Model utilities for fine-tuning project.
"""
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any
import os


class ModelManager:
    """Model management class."""
    
    def __init__(self, config):
        """Initialize model manager."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def load_model_and_tokenizer(self, model_name: str, token: str) -> tuple:
        """Load model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            padding_side=self.config.model.padding_side
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config if enabled
        quantization_config = None
        if self.config.quantization.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.quantization.use_4bit,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=getattr(torch, self.config.quantization.bnb_4bit_compute_dtype)
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        return self.model, self.tokenizer
    
    def setup_lora(self) -> Any:
        """Setup LoRA for the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora.target_modules
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def setup_training_args(self, output_dir: str) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            num_train_epochs=self.config.training.num_epochs,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_steps=10,
            eval_steps=self.config.evaluation.eval_steps,
            save_steps=self.config.evaluation.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["wandb"] if self.config.logging.use_wandb else [],
            run_name=f"finetune-{self.config.model.name.split('/')[-1]}",
            remove_unused_columns=False,
        )
    
    def create_trainer(self, train_dataset, eval_dataset, training_args) -> Trainer:
        """Create trainer."""
        if self.peft_model is None:
            raise ValueError("LoRA model not setup. Call setup_lora() first.")
        
        return Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model."""
        if self.peft_model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(save_path, exist_ok=True)
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_finetuned_model(self, model_path: str, token: str):
        """Load a fine-tuned model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
        self.peft_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        return self.peft_model, self.tokenizer
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate response for a given prompt."""
        if self.peft_model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded.")
        
        # Create messages
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_length
        ).to(self.peft_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        generated_text = response[len(text):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.peft_model is None:
            return {"error": "Model not loaded"}
        
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "model_name": self.config.model.name,
            "device": str(next(self.peft_model.parameters()).device),
            "dtype": str(next(self.peft_model.parameters()).dtype)
        }
