"""
Multi-Config Embedding Generator for SenticCrystal.

Supports multiple high-performance configurations:
- Config 204: simple_mean + sum + lstm (68.0%)
- Config 205: simple_mean + sum + transformer (67.9%) 
- Config 206: simple_mean + sum + context_lstm (69.7%)
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Add paths for original modules
LINGUA_EMOCA_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(LINGUA_EMOCA_DIR / 'src'))

try:
    from sroberta_module import SentenceEmbedder
    from wnaffect_module import WordNetAffectEmbedder
except ImportError:
    logger.warning("Original Lingua-Emoca modules not found, using simplified implementation")
    SentenceEmbedder = None
    WordNetAffectEmbedder = None

class MultiConfigEmbeddingGenerator:
    """
    Generate embeddings for multiple high-performance configurations.
    
    Supported configs:
    - Config 204: simple_mean + sum + lstm
    - Config 205: simple_mean + sum + transformer  
    - Config 206: simple_mean + sum + context_lstm
    """
    
    def __init__(self, config_name: str, device: str = 'cpu', context_turns: int = 4):
        self.config_name = config_name
        self.device = device
        self.context_turns = context_turns
        
        # Get config settings
        self.config = self._get_config_settings(config_name)
        
        logger.info(f"Initializing {config_name} EmbeddingGenerator on {device}")
        logger.info(f"Config: {self.config}")
        
        # Initialize components
        self._init_sentence_embedder()
        self._init_word_embedder()
    
    def _get_config_settings(self, config_name: str) -> Dict[str, Any]:
        """Get settings for specific config."""
        configs = {
            'config204': {
                'apply_word_pe': False,
                'pooling_method': 'simple_mean',
                'apply_sentence_pe': False,
                'combination_method': 'sum',
                'bayesian_method': 'lstm',
                'context_turns': self.context_turns,
                'description': 'Simple mean pooling + sum combination + LSTM'
            },
            'config205': {
                'apply_word_pe': False,
                'pooling_method': 'simple_mean',
                'apply_sentence_pe': False,
                'combination_method': 'sum',
                'bayesian_method': 'transformer',
                'context_turns': self.context_turns,
                'description': 'Simple mean pooling + sum combination + Transformer'
            },
            'config206': {
                'apply_word_pe': False,
                'pooling_method': 'simple_mean',
                'apply_sentence_pe': False,
                'combination_method': 'sum',
                'bayesian_method': 'context_lstm',
                'context_turns': self.context_turns,
                'description': 'Simple mean pooling + sum combination + Context LSTM'
            }
        }
        
        if config_name.lower() not in configs:
            raise ValueError(f"Unsupported config: {config_name}. Supported: {list(configs.keys())}")
        
        return configs[config_name.lower()]
    
    def _init_sentence_embedder(self):
        """Initialize Sentence-RoBERTa embedder."""
        try:
            if SentenceEmbedder is not None:
                self.sentence_embedder = SentenceEmbedder()
                logger.info("✅ Initialized original SentenceEmbedder")
            else:
                raise ImportError("Original module not available")
        except Exception as e:
            logger.warning(f"Using fallback SentenceTransformer: {e}")
            # Fallback to direct SentenceTransformer
            model_name = "all-mpnet-base-v2"  # High-quality alternative
            self.sentence_model = SentenceTransformer(model_name)
            self.sentence_model.to(self.device)
            logger.info(f"✅ Initialized fallback SentenceTransformer: {model_name}")
    
    def _init_word_embedder(self):
        """Initialize WordNet-Affect embedder.""" 
        try:
            if WordNetAffectEmbedder is not None:
                # Initialize with original paths if available
                wn_model_path = LINGUA_EMOCA_DIR / 'src' / 'wn-affect-1.1'
                corpus_path = LINGUA_EMOCA_DIR / 'src' / 'wn-domains-3.2'
                
                if wn_model_path.exists():
                    self.word_embedder = WordNetAffectEmbedder(str(wn_model_path), str(corpus_path))
                    logger.info("✅ Initialized original WordNetAffectEmbedder")
                else:
                    raise FileNotFoundError("WordNet-Affect files not found")
            else:
                raise ImportError("Original module not available")
        except Exception as e:
            logger.warning(f"WordNet-Affect not available: {e}")
            self.word_embedder = None
    
    def _get_sentence_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding."""
        if hasattr(self, 'sentence_embedder'):
            embedding = self.sentence_embedder.get_embedding(text)
        else:
            # Fallback
            embedding = self.sentence_model.encode(text)
        
        # Ensure float32 type for PyTorch compatibility
        return np.array(embedding, dtype=np.float32)
    
    def _get_word_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get WordNet-Affect embedding if available."""
        if self.word_embedder is not None:
            try:
                embedding = self.word_embedder.get_embedding(text)
                if embedding is not None:
                    # Ensure float32 type for PyTorch compatibility
                    return np.array(embedding, dtype=np.float32)
            except Exception:
                return None
        return None
    
    def _apply_pooling(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Apply pooling method."""
        if self.config['pooling_method'] == 'simple_mean':
            # Equal weighting for all context turns
            result = np.mean(embeddings, axis=0)
        elif self.config['pooling_method'] == 'weighted_mean':
            # Weighted towards more recent turns
            weights = np.array([0.1, 0.2, 0.3, 0.4])[:len(embeddings)]
            weights = weights / weights.sum()
            result = np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown pooling method: {self.config['pooling_method']}")
        
        # Ensure float32 type for PyTorch compatibility
        return np.array(result, dtype=np.float32)
    
    def _combine_embeddings(self, sentence_emb: np.ndarray, word_emb: Optional[np.ndarray]) -> np.ndarray:
        """Combine sentence and word embeddings."""
        if word_emb is None:
            result = sentence_emb
        elif self.config['combination_method'] == 'sum':
            result = sentence_emb + word_emb
        elif self.config['combination_method'] == 'concatenate':
            result = np.concatenate([sentence_emb, word_emb])
        else:
            raise ValueError(f"Unknown combination method: {self.config['combination_method']}")
        
        # Ensure float32 type for PyTorch compatibility
        return np.array(result, dtype=np.float32)
    
    def generate_embeddings(self, texts: List[str], ids: List[str], context_turns: int = None) -> List[Tuple[np.ndarray, str]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text utterances
            ids: List of corresponding IDs
            context_turns: Number of context turns to use
            
        Returns:
            List of (embedding, id) tuples
        """
        if context_turns is None:
            context_turns = self.context_turns
        
        logger.info(f"Generating {self.config_name} embeddings for {len(texts)} texts with {context_turns}-turn context")
        
        results = []
        
        for i, (text, text_id) in enumerate(zip(texts, ids)):
            # Get context window
            start_idx = max(0, i - context_turns + 1)
            context_texts = texts[start_idx:i+1]
            
            # Generate embeddings for context
            context_embeddings = []
            for ctx_text in context_texts:
                # Get sentence embedding
                sent_emb = self._get_sentence_embedding(ctx_text)
                
                # Get word embedding (optional)
                word_emb = self._get_word_embedding(ctx_text)
                
                # Combine embeddings
                combined_emb = self._combine_embeddings(sent_emb, word_emb)
                context_embeddings.append(combined_emb)
            
            # Apply pooling to get final embedding
            final_embedding = self._apply_pooling(context_embeddings)
            
            results.append((final_embedding, text_id))
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(texts)} texts")
        
        logger.info(f"✅ Generated {len(results)} {self.config_name} embeddings")
        return results

# Convenience classes for each config
class Config204EmbeddingGenerator(MultiConfigEmbeddingGenerator):
    def __init__(self, device: str = 'cpu', context_turns: int = 4):
        super().__init__('config204', device, context_turns)

class Config205EmbeddingGenerator(MultiConfigEmbeddingGenerator):
    def __init__(self, device: str = 'cpu', context_turns: int = 4):
        super().__init__('config205', device, context_turns)

class Config206EmbeddingGenerator(MultiConfigEmbeddingGenerator):
    def __init__(self, device: str = 'cpu', context_turns: int = 4):
        super().__init__('config206', device, context_turns)