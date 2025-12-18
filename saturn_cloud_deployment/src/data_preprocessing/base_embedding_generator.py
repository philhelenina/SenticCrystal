"""
Base Embedding Generator for SenticCrystal.

Abstract base class that provides common functionality for all embedding generators.
Extracts shared logic from multiple generator implementations to reduce code duplication.
"""

import sys
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import logging
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseEmbeddingGenerator(ABC):
    """
    Abstract base class for all embedding generators.
    
    Provides common functionality:
    - Model initialization (S-RoBERTa + WordNet-Affect)
    - Context window creation (K-turn logic)
    - Configuration management (Config146 parameters)
    - File I/O operations
    - Error handling and logging
    """
    
    def __init__(self, device: str = 'cpu', config: Optional[Dict[str, Any]] = None):
        """
        Initialize base embedding generator.
        
        Args:
            device: Computing device ('cpu', 'cuda', 'mps')
            config: Configuration dictionary with embedding parameters
        """
        self.device = device
        self.config = self._load_default_config(config)
        
        # Initialize models
        self.sentence_embedder = None
        self.wn_embedder = None
        
        logger.info(f"Initializing {self.__class__.__name__} on device: {device}")
        self._initialize_models()
    
    def _load_default_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load default Config146 configuration.
        
        Args:
            config: User-provided configuration (optional)
            
        Returns:
            Merged configuration with defaults
        """
        # Config146 optimal settings (72.1% Macro-F1)
        default_config = {
            'apply_word_pe': False,
            'pooling_method': 'weighted_mean', 
            'apply_sentence_pe': False,
            'combination_method': 'sum',
            'bayesian_method': 'context_lstm',
            'context_turns': 6,  # Default K value
            'alpha': 0.5,  # Combination weight
            'sentence_model': 'nli-distilroberta-base-v2',  # Config146 actual model
            'wordnet_model_path': 'scripts/wn-affect-1.0',
            'embedding_dim': 768  # S-RoBERTa (768) + WN-Affect (300) → sum → 768
        }
        
        if config:
            default_config.update(config)
            
        return default_config
    
    def _initialize_models(self):
        """Initialize Sentence-RoBERTa and WordNet-Affect models."""
        try:
            # Initialize Sentence-RoBERTa
            self._initialize_sentence_model()
            
            # Initialize WordNet-Affect  
            self._initialize_wordnet_model()
            
            logger.info("✅ Model initialization completed")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
    
    def _initialize_sentence_model(self):
        """Initialize Sentence-RoBERTa model."""
        try:
            model_name = self.config['sentence_model']
            logger.info(f"🤗 Loading Sentence-RoBERTa: {model_name}")
            
            self.sentence_model = SentenceTransformer(model_name, device=self.device)
            
            # Verify embedding dimension
            test_embedding = self.sentence_model.encode("test")
            actual_dim = len(test_embedding)
            logger.info(f"📏 S-RoBERTa embedding dimension: {actual_dim}")
            
            # Try to load custom SentenceEmbedder if available
            self._try_load_custom_sentence_embedder()
            
        except Exception as e:
            logger.warning(f"Failed to load Sentence-RoBERTa: {str(e)}")
            raise
    
    def _try_load_custom_sentence_embedder(self):
        """Try to load custom SentenceEmbedder from original codebase."""
        try:
            # Add paths for original modules
            lingua_emoca_dir = Path(__file__).parent.parent.parent.parent
            sys.path.append(str(lingua_emoca_dir / 'src'))
            
            from sroberta_module import SentenceEmbedder
            
            self.sentence_embedder = SentenceEmbedder(
                device=self.device,
                pooling_method=self.config['pooling_method'],
                apply_word_pe=self.config['apply_word_pe'],
                apply_sentence_pe=self.config['apply_sentence_pe']
            )
            
            logger.info("✅ Loaded custom SentenceEmbedder")
            
        except ImportError:
            logger.info("ℹ️ Custom SentenceEmbedder not found, using standard SentenceTransformer")
    
    def _initialize_wordnet_model(self):
        """Initialize WordNet-Affect model."""
        try:
            # Try to load custom WordNet-Affect embedder
            sys.path.append(str(Path(__file__).parent.parent / 'features'))
            from wnaffect_module import WordNetAffectEmbedder
            
            wn_path = self.config['wordnet_model_path']
            logger.info(f"📚 Loading WordNet-Affect from: {wn_path}")
            
            # WordNet-Affect embedder expects (wn_model, corpus_path) parameters
            # We'll initialize with minimal setup for now
            try:
                import gensim.downloader as api
                wn_model = api.load('word2vec-google-news-300')
                corpus_path = f"{wn_path}/a-synsets.xml"
                
                self.wn_embedder = WordNetAffectEmbedder(
                    wn_model=wn_model,
                    corpus_path=corpus_path
                )
                
                logger.info("✅ WordNet-Affect loaded successfully")
                
            except Exception as model_error:
                logger.warning(f"⚠️ WordNet-Affect model loading failed: {model_error}")
                logger.info("ℹ️ Continuing without WordNet-Affect features")
                self.wn_embedder = None
            
        except ImportError as import_error:
            logger.warning(f"⚠️ WordNet-Affect module not found: {import_error}")
            self.wn_embedder = None
    
    def _create_context_window(
        self, 
        embeddings: List[np.ndarray], 
        ids: List[str], 
        context_turns: Optional[int] = None,
        dialogue_ids: Optional[List[str]] = None
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Create context windows for embeddings with dialogue boundary awareness.
        
        Args:
            embeddings: List of embedding vectors
            ids: List of utterance IDs
            context_turns: Number of turns to include (None = use config default)
            dialogue_ids: List of dialogue IDs for boundary detection
            
        Returns:
            List of (windowed_embedding, utterance_id) tuples
        """
        if context_turns is None:
            context_turns = self.config['context_turns']
            
        windowed_embeddings = []
        
        for i, (emb, sample_id) in enumerate(zip(embeddings, ids)):
            # Determine context window start
            start_idx = max(0, i - context_turns + 1)
            
            # If dialogue IDs provided, respect dialogue boundaries
            if dialogue_ids is not None:
                current_dialogue = dialogue_ids[i]
                # Find dialogue start
                dialogue_start = i
                while dialogue_start > 0 and dialogue_ids[dialogue_start - 1] == current_dialogue:
                    dialogue_start -= 1
                # Don't go beyond dialogue boundary
                start_idx = max(start_idx, dialogue_start)
            
            # Create context window
            context_window = []
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
            
            # Log progress
            if (i + 1) % 1000 == 0:
                logger.debug(f"Created context windows: {i + 1}/{len(embeddings)}")
        
        return windowed_embeddings
    
    def _combine_embeddings(
        self, 
        sentence_emb: np.ndarray, 
        wordnet_emb: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Combine Sentence-RoBERTa and WordNet-Affect embeddings.
        
        Args:
            sentence_emb: S-RoBERTa embedding (768-dim)
            wordnet_emb: WordNet-Affect embedding (300-dim) or None
            
        Returns:
            Combined embedding according to combination_method
        """
        if wordnet_emb is None:
            # No WordNet embedding available, return S-RoBERTa only
            return sentence_emb
            
        method = self.config['combination_method']
        alpha = self.config.get('alpha', 0.5)
        
        if method == 'sum':
            # Resize WordNet to match S-RoBERTa dimension
            if len(wordnet_emb) != len(sentence_emb):
                # Pad or truncate to match S-RoBERTa dimension
                if len(wordnet_emb) < len(sentence_emb):
                    padded_wn = np.zeros(len(sentence_emb))
                    padded_wn[:len(wordnet_emb)] = wordnet_emb
                    wordnet_emb = padded_wn
                else:
                    wordnet_emb = wordnet_emb[:len(sentence_emb)]
            
            return sentence_emb + alpha * wordnet_emb
            
        elif method == 'concatenate':
            return np.concatenate([sentence_emb, wordnet_emb])
            
        elif method == 'weighted_sum':
            # Similar to sum but with explicit weighting
            return (1 - alpha) * sentence_emb + alpha * wordnet_emb
            
        else:
            logger.warning(f"Unknown combination method: {method}, using sum")
            return sentence_emb + wordnet_emb
    
    def save_embeddings(
        self, 
        embeddings: List[Tuple[np.ndarray, str]], 
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save embeddings to file with metadata.
        
        Args:
            embeddings: List of (embedding, id) tuples
            output_path: Output file path
            metadata: Additional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data structure
        save_data = {
            'embeddings': embeddings,
            'config': self.config,
            'metadata': metadata or {},
            'generated_at': datetime.now().isoformat(),
            'generator_class': self.__class__.__name__
        }
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✅ Saved {len(embeddings)} embeddings to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, input_path: Union[str, Path]) -> Tuple[List[Tuple[np.ndarray, str]], Dict[str, Any]]:
        """
        Load embeddings from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        input_path = Path(input_path)
        
        try:
            with open(input_path, 'rb') as f:
                save_data = pickle.load(f)
            
            embeddings = save_data['embeddings']
            metadata = save_data.get('metadata', {})
            
            logger.info(f"✅ Loaded {len(embeddings)} embeddings from {input_path}")
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {str(e)}")
            raise
    
    @abstractmethod
    def generate_embeddings(
        self, 
        texts: List[str], 
        ids: List[str], 
        **kwargs
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate embeddings for given texts.
        
        This is an abstract method that must be implemented by subclasses.
        
        Args:
            texts: List of input texts
            ids: List of utterance IDs
            **kwargs: Additional parameters specific to implementation
            
        Returns:
            List of (embedding, id) tuples
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return (f"{self.__class__.__name__}("
                f"device='{self.device}', "
                f"config={self.config})")