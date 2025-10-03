"""
Configuration management for fine-tuning project.
"""
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str
    max_length: int
    padding_side: str


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float


@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: list


@dataclass
class QuantizationConfig:
    """Quantization configuration parameters."""
    use_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_split: float
    val_split: float
    test_split: float
    max_samples: Optional[int]


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: list
    eval_steps: int
    save_steps: int


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    project_name: str
    log_level: str
    use_wandb: bool
    use_tensorboard: bool


@dataclass
class PathsConfig:
    """Paths configuration parameters."""
    data_path: str
    model_save_path: str
    output_path: str
    cache_dir: str


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self._load_config()
        self._setup_paths()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
            
            # Create configuration objects
            self.model = ModelConfig(**config_dict['model'])
            self.training = TrainingConfig(**config_dict['training'])
            self.lora = LoRAConfig(**config_dict['lora'])
            self.quantization = QuantizationConfig(**config_dict['quantization'])
            self.data = DataConfig(**config_dict['data'])
            self.evaluation = EvaluationConfig(**config_dict['evaluation'])
            self.logging = LoggingConfig(**config_dict['logging'])
            self.paths = PathsConfig(**config_dict['paths'])
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _setup_paths(self):
        """Create necessary directories."""
        for path_attr in ['data_path', 'model_save_path', 'output_path', 'cache_dir']:
            path = getattr(self.paths, path_attr)
            os.makedirs(path, exist_ok=True)
    
    def get_hf_token(self) -> str:
        """Get Hugging Face token from environment variable."""
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN environment variable not set")
        return token
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'lora': self.lora.__dict__,
            'quantization': self.quantization.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__,
            'logging': self.logging.__dict__,
            'paths': self.paths.__dict__
        }
