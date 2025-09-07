# src/utils/config_manager.py
"""
ConfigManager: Unified configuration management system for SenticCrystal
Manages 200+ experimental configurations in a single, organized system
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str  # 'lstm', 'mlp', 'transformer', 'ensemble'
    input_dim: int
    hidden_dims: List[int]
    num_classes: int = 4
    dropout: float = 0.3
    activation: str = 'relu'
    batch_norm: bool = True

@dataclass
class FocalLossConfig:
    """Focal loss configuration"""
    use_focal_loss: bool = False
    alpha: float = 1.0
    gamma: float = 1.2
    class_weights: Optional[List[float]] = None

@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    apply_word_pe: bool = False
    pooling_method: str = 'lstm'  # 'lstm', 'self_attention', 'weighted_mean', 'simple_mean'
    apply_sentence_pe: bool = False
    combination_method: str = 'sum'  # 'sum', 'concatenate', 'cross_attention_*'
    alpha_range: Optional[List[float]] = None
    context_window: int = 5  # number of turns

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 100
    early_stopping_patience: int = 10
    optimizer: str = 'adam'
    scheduler: str = 'reduce_on_plateau'
    gradient_clip: float = 1.0

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    id: int
    name: str
    description: str
    model: ModelConfig
    focal_loss: FocalLossConfig
    embedding: EmbeddingConfig
    training: TrainingConfig
    created_at: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ConfigManager:
    """
    Unified configuration management system
    Replaces 200+ individual config files with a single, organized system
    """
    
    def __init__(self, base_path: str = "configs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded configurations
        self.config_cache: Dict[int, ExperimentConfig] = {}
        
        # Load existing configurations.json if available
        self.legacy_configs = self._load_legacy_configs()
        
        # Define configuration patterns based on ID ranges
        self.config_patterns = {
            (1, 50): self._generate_lstm_config,
            (51, 100): self._generate_mlp_config,
            (101, 146): self._generate_roberta_config,
            (147, 175): self._generate_ensemble_config,
            (176, 209): self._generate_focal_variants
        }
    
    def _load_legacy_configs(self) -> Dict[int, Dict]:
        """Load existing configurations.json file"""
        legacy_path = Path("configurations.json")
        if legacy_path.exists():
            with open(legacy_path, 'r') as f:
                data = json.load(f)
                return {config['id']: config for config in data.get('configurations', [])}
        return {}
    
    def get_config(self, config_id: int) -> ExperimentConfig:
        """
        Get configuration by ID
        First checks cache, then legacy configs, then generates new
        """
        # Check cache
        if config_id in self.config_cache:
            return self.config_cache[config_id]
        
        # Check legacy configurations
        if config_id in self.legacy_configs:
            config = self._convert_legacy_config(config_id, self.legacy_configs[config_id])
            self.config_cache[config_id] = config
            return config
        
        # Generate new configuration based on patterns
        config = self._generate_config(config_id)
        self.config_cache[config_id] = config
        return config
    
    def _convert_legacy_config(self, config_id: int, legacy: Dict) -> ExperimentConfig:
        """Convert legacy configuration format to new format"""
        # Determine input dimensions based on configuration
        if legacy.get('combination_method') == 'concatenate':
            input_dim = 1536  # 768 * 2
        else:
            input_dim = 768
        
        # Multiply by context window size
        context_window = 5  # default
        total_input_dim = input_dim * context_window
        
        return ExperimentConfig(
            id=config_id,
            name=f"Config_{config_id}",
            description=legacy.get('description', f"Legacy config {config_id}"),
            model=ModelConfig(
                model_type='mlp',
                input_dim=total_input_dim,
                hidden_dims=[512, 256]
            ),
            focal_loss=FocalLossConfig(),
            embedding=EmbeddingConfig(
                apply_word_pe=legacy.get('apply_word_pe', False),
                pooling_method=legacy.get('pooling_method', 'lstm'),
                apply_sentence_pe=legacy.get('apply_sentence_pe', False),
                combination_method=legacy.get('combination_method', 'sum'),
                alpha_range=legacy.get('alpha_range', None)
            ),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def _generate_config(self, config_id: int) -> ExperimentConfig:
        """Generate configuration based on ID patterns"""
        for (start, end), generator in self.config_patterns.items():
            if start <= config_id <= end:
                return generator(config_id)
        
        # Default configuration for undefined ranges
        return self._generate_default_config(config_id)
    
    def _generate_lstm_config(self, config_id: int) -> ExperimentConfig:
        """Generate LSTM-based configurations (ID 1-50)"""
        # Vary context window and hidden sizes
        context_windows = [2, 3, 4, 5]
        hidden_sizes = [[256], [512], [256, 128], [512, 256]]
        
        context_window = context_windows[config_id % 4]
        hidden_dims = hidden_sizes[(config_id // 4) % 4]
        
        return ExperimentConfig(
            id=config_id,
            name=f"LSTM_Context{context_window}_Config{config_id}",
            description=f"LSTM with {context_window}-turn context, hidden dims {hidden_dims}",
            model=ModelConfig(
                model_type='lstm',
                input_dim=768 * context_window,
                hidden_dims=hidden_dims
            ),
            focal_loss=FocalLossConfig(use_focal_loss=False),
            embedding=EmbeddingConfig(
                pooling_method='lstm',
                context_window=context_window
            ),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def _generate_mlp_config(self, config_id: int) -> ExperimentConfig:
        """Generate MLP-based configurations (ID 51-100)"""
        # Vary hidden dimensions and dropout
        hidden_options = [
            [512, 256],
            [256, 128],
            [512, 256, 128],
            [1024, 512, 256]
        ]
        dropout_options = [0.2, 0.3, 0.4, 0.5]
        
        hidden_dims = hidden_options[(config_id - 51) % 4]
        dropout = dropout_options[(config_id - 51) // 4 % 4]
        
        return ExperimentConfig(
            id=config_id,
            name=f"MLP_Config{config_id}",
            description=f"MLP with hidden dims {hidden_dims}, dropout {dropout}",
            model=ModelConfig(
                model_type='mlp',
                input_dim=768 * 5,  # 5-turn context
                hidden_dims=hidden_dims,
                dropout=dropout
            ),
            focal_loss=FocalLossConfig(use_focal_loss=False),
            embedding=EmbeddingConfig(),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def _generate_roberta_config(self, config_id: int) -> ExperimentConfig:
        """Generate RoBERTa-based configurations (ID 101-146)"""
        # Config 146 is special - it's your best performing model
        if config_id == 146:
            return ExperimentConfig(
                id=146,
                name="RoBERTa_Optimal_Config146",
                description="Best performing RoBERTa configuration with 4-turn context",
                model=ModelConfig(
                    model_type='mlp',
                    input_dim=768 * 5,  # 5-turn but weighted
                    hidden_dims=[512, 256]
                ),
                focal_loss=FocalLossConfig(
                    use_focal_loss=True,
                    alpha=1.0,
                    gamma=1.2
                ),
                embedding=EmbeddingConfig(
                    pooling_method='weighted_mean',
                    context_window=5
                ),
                training=TrainingConfig(),
                created_at=datetime.now().isoformat()
            )
        
        # Other RoBERTa variations
        pooling_methods = ['lstm', 'self_attention', 'weighted_mean', 'simple_mean']
        pooling = pooling_methods[(config_id - 101) % 4]
        
        return ExperimentConfig(
            id=config_id,
            name=f"RoBERTa_{pooling}_Config{config_id}",
            description=f"RoBERTa with {pooling} pooling",
            model=ModelConfig(
                model_type='mlp',
                input_dim=768 * 5,
                hidden_dims=[512, 256]
            ),
            focal_loss=FocalLossConfig(use_focal_loss=False),
            embedding=EmbeddingConfig(pooling_method=pooling),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def _generate_ensemble_config(self, config_id: int) -> ExperimentConfig:
        """Generate ensemble configurations (ID 147-175)"""
        return ExperimentConfig(
            id=config_id,
            name=f"Ensemble_Config{config_id}",
            description=f"Ensemble of models {config_id % 3 + 1} components",
            model=ModelConfig(
                model_type='ensemble',
                input_dim=768 * 5,
                hidden_dims=[512, 256]
            ),
            focal_loss=FocalLossConfig(use_focal_loss=True, alpha=1.0, gamma=1.2),
            embedding=EmbeddingConfig(),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def _generate_focal_variants(self, config_id: int) -> ExperimentConfig:
        """Generate Focal Loss variants (ID 176-209)"""
        # Vary alpha and gamma systematically
        alpha_values = [0.8, 0.9, 1.0, 1.1, 1.2]
        gamma_values = [0.8, 1.0, 1.2, 1.5, 2.0]
        
        idx = config_id - 176
        alpha = alpha_values[idx % 5]
        gamma = gamma_values[idx // 5 % 5]
        
        return ExperimentConfig(
            id=config_id,
            name=f"FocalLoss_a{alpha}_g{gamma}_Config{config_id}",
            description=f"Focal Loss experiment with alpha={alpha}, gamma={gamma}",
            model=ModelConfig(
                model_type='mlp',
                input_dim=768 * 5,
                hidden_dims=[512, 256]
            ),
            focal_loss=FocalLossConfig(
                use_focal_loss=True,
                alpha=alpha,
                gamma=gamma
            ),
            embedding=EmbeddingConfig(),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def _generate_default_config(self, config_id: int) -> ExperimentConfig:
        """Generate default configuration for undefined IDs"""
        return ExperimentConfig(
            id=config_id,
            name=f"Default_Config{config_id}",
            description=f"Default configuration for ID {config_id}",
            model=ModelConfig(
                model_type='mlp',
                input_dim=768 * 5,
                hidden_dims=[512, 256]
            ),
            focal_loss=FocalLossConfig(),
            embedding=EmbeddingConfig(),
            training=TrainingConfig(),
            created_at=datetime.now().isoformat()
        )
    
    def save_config(self, config: ExperimentConfig, format: str = 'yaml') -> Path:
        """Save configuration to file"""
        # Create subdirectory based on config type
        subdir = self.base_path / config.model.model_type
        subdir.mkdir(parents=True, exist_ok=True)
        
        filename = f"config_{config.id}_{config.get_hash()}"
        
        if format == 'yaml':
            filepath = subdir / f"{filename}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:  # json
            filepath = subdir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        
        return filepath
    
    def load_config_from_file(self, filepath: Path) -> ExperimentConfig:
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            if filepath.suffix == '.yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert nested dictionaries to dataclasses
        return ExperimentConfig(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            model=ModelConfig(**data['model']),
            focal_loss=FocalLossConfig(**data['focal_loss']),
            embedding=EmbeddingConfig(**data['embedding']),
            training=TrainingConfig(**data['training']),
            created_at=data['created_at'],
            version=data.get('version', '1.0')
        )
    
    def list_configs(self, model_type: Optional[str] = None) -> List[Tuple[int, str]]:
        """List all available configurations"""
        configs = []
        
        # Add all possible config IDs with descriptions
        for config_id in range(1, 210):
            config = self.get_config(config_id)
            if model_type is None or config.model.model_type == model_type:
                configs.append((config_id, config.name))
        
        return configs
    
    def export_all_configs(self, output_dir: Path, format: str = 'yaml'):
        """Export all configurations to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = []
        
        for config_id in range(1, 210):
            config = self.get_config(config_id)
            filepath = self.save_config(config, format)
            
            summary.append({
                'id': config_id,
                'name': config.name,
                'file': str(filepath),
                'hash': config.get_hash()
            })
            
            print(f"Exported config {config_id}: {filepath}")
        
        # Save summary
        summary_file = output_dir / "config_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExported {len(summary)} configurations")
        print(f"Summary saved to: {summary_file}")
    
    def get_best_configs(self, top_k: int = 5) -> List[ExperimentConfig]:
        """Get best performing configurations based on known results"""
        # Based on your experimental results
        best_config_ids = [146, 1, 2, 147, 150]  # Add more based on your results
        
        configs = []
        for config_id in best_config_ids[:top_k]:
            configs.append(self.get_config(config_id))
        
        return configs
    
    def create_experiment_group(self, 
                               name: str,
                               base_config_id: int,
                               variations: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """
        Create a group of experiments based on a base configuration with variations
        
        Example:
            variations = {
                'focal_loss.alpha': [0.8, 1.0, 1.2],
                'focal_loss.gamma': [1.0, 1.2, 1.5]
            }
        """
        base_config = self.get_config(base_config_id)
        experiment_configs = []
        
        # Generate all combinations
        import itertools
        keys = list(variations.keys())
        values = list(variations.values())
        
        for combination in itertools.product(*values):
            # Create new config based on base
            new_config = self._copy_config(base_config)
            
            # Apply variations
            for key, value in zip(keys, combination):
                self._set_nested_attr(new_config, key, value)
            
            # Update metadata
            new_config.id = len(self.config_cache) + 1000  # New ID range for custom configs
            new_config.name = f"{name}_experiment_{len(experiment_configs)}"
            new_config.created_at = datetime.now().isoformat()
            
            experiment_configs.append(new_config)
            self.config_cache[new_config.id] = new_config
        
        return experiment_configs
    
    def _copy_config(self, config: ExperimentConfig) -> ExperimentConfig:
        """Deep copy a configuration"""
        import copy
        return copy.deepcopy(config)
    
    def _set_nested_attr(self, obj: Any, path: str, value: Any):
        """Set nested attribute using dot notation"""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


# Example usage and migration script
if __name__ == "__main__":
    # Initialize ConfigManager
    manager = ConfigManager()
    
    # Example: Get configuration 146 (our best model)
    config_146 = manager.get_config(146)
    print(f"Config 146: {config_146.name}")
    print(f"  Model type: {config_146.model.model_type}")
    print(f"  Input dim: {config_146.model.input_dim}")
    print(f"  Focal Loss: α={config_146.focal_loss.alpha}, γ={config_146.focal_loss.gamma}")
    
    # Example: Create new experiment group
    focal_experiments = manager.create_experiment_group(
        name="focal_loss_optimization",
        base_config_id=146,
        variations={
            'focal_loss.alpha': [0.8, 1.0, 1.2],
            'focal_loss.gamma': [1.0, 1.2, 1.5]
        }
    )
    print(f"\nCreated {len(focal_experiments)} focal loss experiments")
    
    # Example: Export all configurations
    # manager.export_all_configs(Path("configs/exported"), format='yaml')
    
    # Example: List best configurations
    best_configs = manager.get_best_configs(top_k=3)
    print("\nTop 3 configurations:")
    for config in best_configs:
        print(f"  {config.id}: {config.name}")
