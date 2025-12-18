"""
Config146 Embedding Generator for SenticCrystal.

Generates embeddings using the Config146 architecture that achieved 72.1% on IEMOCAP.
Combines WordNet-Affect + Sentence-RoBERTa with 4-turn context modeling.
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
except ImportError:
    SentenceEmbedder = None

try:
    sys.path.append(str(Path(__file__).parent.parent / 'features'))
    from wnaffect_module import WordNetAffectEmbedder
except ImportError:
    logger.warning("WordNet-Affect module not found")
    WordNetAffectEmbedder = None

class Config146EmbeddingGenerator:
    """
    Generate Config146-style embeddings for emotion recognition.
    
    Config146 achieved 72.1% accuracy with:
    - apply_word_pe: False
    - pooling_method: "weighted_mean"
    - apply_sentence_pe: False  
    - combination_method: "sum"
    - bayesian_method: "context_lstm"
    - context_turns: 4
    """
    
    def __init__(self, device: str = 'cpu', config: Dict[str, Any] = None):
        self.device = device
        self.config = config or self._get_default_config()
        
        logger.info(f"Initializing Config146 EmbeddingGenerator on {device}")
        logger.info(f"Config: {self.config}")
        
        # Initialize components
        self._init_sentence_embedder()
        self._init_word_embedder()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Config146 settings."""
        return {
            'apply_word_pe': False,
            'pooling_method': 'weighted_mean',
            'apply_sentence_pe': False,
            'combination_method': 'sum',
            'bayesian_method': 'context_lstm',
            'context_turns': 4
        }
    
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
                # Initialize with local paths
                base_path = Path(__file__).parent.parent.parent
                wn_model_path = base_path / 'scripts' / 'wn-domains-3.2' / 'wn-affect-1.1'
                corpus_path = base_path / 'scripts' / 'wn-domains-3.2' / 'wn-affect-1.1'
                
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
            logger.info("⚠️  WordNet-Affect disabled - using Sentence-RoBERTa only")
    
    def _get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings."""
        if hasattr(self, 'sentence_embedder'):
            # Use original SentenceEmbedder
            embeddings = []
            for text in texts:
                emb = self.sentence_embedder.get_sentence_embedding(text)
                embeddings.append(emb)
            return np.array(embeddings)
        else:
            # Use fallback SentenceTransformer
            embeddings = self.sentence_model.encode(texts, convert_to_numpy=True)
            return embeddings
    
    def _get_word_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate word embeddings using WordNet-Affect."""
        if self.word_embedder is None:
            return None
            
        embeddings = []
        for text in texts:
            try:
                emb = self.word_embedder.get_word_embedding(text)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"WordNet embedding failed for text: {e}")
                # Fallback to zero vector
                embeddings.append(np.zeros(300))  # WordNet-Affect uses 300-dim
        
        return np.array(embeddings)
    
    def _combine_embeddings(self, sentence_emb: np.ndarray, word_emb: Optional[np.ndarray] = None) -> np.ndarray:
        """Combine sentence and word embeddings using Config146 method."""
        if word_emb is None:
            return sentence_emb
        
        combination_method = self.config['combination_method']
        
        if combination_method == 'sum':
            # Pad to same dimension if needed
            if sentence_emb.shape[-1] != word_emb.shape[-1]:
                min_dim = min(sentence_emb.shape[-1], word_emb.shape[-1])
                sentence_emb = sentence_emb[..., :min_dim]
                word_emb = word_emb[..., :min_dim]
            
            return sentence_emb + word_emb
        
        elif combination_method == 'concatenate':
            return np.concatenate([sentence_emb, word_emb], axis=-1)
        
        else:
            logger.warning(f"Unknown combination method: {combination_method}, using sum")
            return sentence_emb + word_emb
    
    def _apply_pooling(self, embeddings: np.ndarray, method: str = 'weighted_mean') -> np.ndarray:
        """Apply pooling method to embeddings."""
        if method == 'weighted_mean':
            # Simple weighted mean (in practice, weights would be learned)
            # For now, use equal weights
            weights = np.ones(embeddings.shape[0]) / embeddings.shape[0]
            return np.average(embeddings, axis=0, weights=weights)
        
        elif method == 'simple_mean':
            return np.mean(embeddings, axis=0)
        
        elif method == 'max':
            return np.max(embeddings, axis=0)
        
        else:
            logger.warning(f"Unknown pooling method: {method}, using simple_mean")
            return np.mean(embeddings, axis=0)
    
    def _create_context_window(self, embeddings: List[np.ndarray], ids: List[str], context_turns: int = 4) -> List[Tuple[np.ndarray, str]]:
        """Create context windows for embeddings."""
        windowed_embeddings = []
        
        for i, (emb, sample_id) in enumerate(zip(embeddings, ids)):
            # Create context window
            context_window = []
            
            # Get previous context (up to context_turns-1 previous samples)
            start_idx = max(0, i - context_turns + 1)
            for j in range(start_idx, i + 1):
                context_window.append(embeddings[j])
            
            # Pad if necessary to ensure consistent context_turns length
            while len(context_window) < context_turns:
                # Pad with zeros at the beginning
                context_window.insert(0, np.zeros_like(emb))
            
            # Take only the last context_turns embeddings
            context_window = context_window[-context_turns:]
            
            # Stack into (context_turns, embedding_dim) format
            windowed_emb = np.stack(context_window)
            windowed_embeddings.append((windowed_emb, sample_id))
        
        return windowed_embeddings
    
    def generate_embeddings(self, texts: List[str], ids: List[str], context_turns: int = 4) -> List[Tuple[np.ndarray, str]]:
        """
        Generate Config146-style embeddings with context windows.
        
        Args:
            texts: List of text strings
            ids: List of sample IDs  
            context_turns: Number of context turns (default: 4)
            
        Returns:
            List of (embedding, id) tuples where embedding has shape (context_turns, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts with {context_turns}-turn context...")
        
        # Generate sentence embeddings
        logger.info("🔄 Generating sentence embeddings...")
        sentence_embeddings = self._get_sentence_embeddings(texts)
        logger.info(f"✅ Sentence embeddings shape: {sentence_embeddings.shape}")
        
        # Generate word embeddings if available
        word_embeddings = None
        if self.word_embedder is not None:
            logger.info("🔄 Generating word embeddings...")
            word_embeddings = self._get_word_embeddings(texts)
            logger.info(f"✅ Word embeddings shape: {word_embeddings.shape}")
        
        # Combine embeddings
        logger.info("🔄 Combining embeddings...")
        combined_embeddings = []
        for i in range(len(texts)):
            sent_emb = sentence_embeddings[i]
            word_emb = word_embeddings[i] if word_embeddings is not None else None
            combined = self._combine_embeddings(sent_emb, word_emb)
            combined_embeddings.append(combined)
        
        logger.info(f"✅ Combined embeddings shape: {len(combined_embeddings)} x {combined_embeddings[0].shape}")
        
        # Create context windows
        logger.info(f"🔄 Creating {context_turns}-turn context windows...")
        windowed_embeddings = self._create_context_window(combined_embeddings, ids, context_turns)
        
        logger.info(f"✅ Generated {len(windowed_embeddings)} context windows")
        if len(windowed_embeddings) > 0:
            logger.info(f"✅ Context window shape: {windowed_embeddings[0][0].shape}")
        
        return windowed_embeddings

def test_embedding_generator():
    """Test the embedding generator."""
    logger.info("Testing Config146 EmbeddingGenerator...")
    
    # Setup
    device = 'cpu'  # Use CPU for testing
    config = {
        'apply_word_pe': False,
        'pooling_method': 'weighted_mean',
        'apply_sentence_pe': False,
        'combination_method': 'sum', 
        'bayesian_method': 'context_lstm',
        'context_turns': 4
    }
    
    # Test data
    texts = [
        "I am very happy today!",
        "This makes me feel sad.",
        "I'm getting angry about this.",
        "Everything seems neutral to me.",
        "What an exciting day!"
    ]
    ids = [f"test_{i}" for i in range(len(texts))]
    
    # Generate embeddings
    generator = Config146EmbeddingGenerator(device=device, config=config)
    embeddings = generator.generate_embeddings(texts, ids, context_turns=4)
    
    # Verify results
    logger.info(f"✅ Generated {len(embeddings)} embeddings")
    for i, (emb, sample_id) in enumerate(embeddings):
        logger.info(f"Sample {i}: ID={sample_id}, shape={emb.shape}")
    
    logger.info("🎉 Test completed successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_embedding_generator()