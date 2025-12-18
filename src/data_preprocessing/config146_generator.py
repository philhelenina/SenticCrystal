"""
Config146 Embedding Generator for SenticCrystal.

Refactored implementation using BaseEmbeddingGenerator.
Generates embeddings using the Config146 architecture that achieved 72.1% on IEMOCAP.
Combines WordNet-Affect + Sentence-RoBERTa with dynamic K-turn context modeling.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import logging

# Import base class
from .base_embedding_generator import BaseEmbeddingGenerator

logger = logging.getLogger(__name__)


class Config146EmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Config146 Embedding Generator.
    
    Implements the Config146 architecture that achieved 72.1% Macro-F1:
    - apply_word_pe: False
    - pooling_method: "weighted_mean"
    - apply_sentence_pe: False  
    - combination_method: "sum"
    - bayesian_method: "context_lstm"
    - Dynamic context_turns (not fixed at 6)
    """
    
    def __init__(self, device: str = 'cpu', config: Optional[Dict[str, Any]] = None):
        """
        Initialize Config146 embedding generator.
        
        Args:
            device: Computing device ('cpu', 'cuda', 'mps')
            config: Configuration dictionary (uses Config146 defaults)
        """
        super().__init__(device=device, config=config)
        
        # Initialize Config146-specific components
        self._initialize_config146_components()
    
    def _initialize_config146_components(self):
        """Initialize Config146-specific components."""
        try:
            # Log Config146 settings
            logger.info("🔧 Config146 Settings:")
            for key, value in self.config.items():
                logger.info(f"  {key}: {value}")
                
            # Additional initialization if needed
            self._setup_pooling_method()
            
        except Exception as e:
            logger.error(f"Failed to initialize Config146 components: {str(e)}")
            raise
    
    def _setup_pooling_method(self):
        """Setup pooling method based on configuration."""
        pooling_method = self.config['pooling_method']
        
        if pooling_method == 'weighted_mean':
            # Config146 optimal setting
            logger.info("✅ Using weighted_mean pooling (Config146 optimal)")
        elif pooling_method == 'mean':
            logger.info("ℹ️ Using mean pooling")
        elif pooling_method == 'max':
            logger.info("ℹ️ Using max pooling")
        else:
            logger.warning(f"Unknown pooling method: {pooling_method}, falling back to weighted_mean")
            self.config['pooling_method'] = 'weighted_mean'
    
    def _get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate sentence embeddings using S-RoBERTa.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of sentence embeddings
        """
        if self.sentence_embedder is not None:
            # Use custom SentenceEmbedder if available
            embeddings = []
            for text in texts:
                try:
                    # Process with custom embedder (includes Config146 settings)
                    emb = self.sentence_embedder.encode(text)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to encode text with custom embedder: {str(e)}")
                    # Fallback to standard model
                    emb = self.sentence_model.encode(text)
                    embeddings.append(emb)
            
            return np.array(embeddings)
        
        else:
            # Use standard SentenceTransformer
            logger.info("Using standard SentenceTransformer")
            embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            return embeddings
    
    def _get_word_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate WordNet-Affect embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of word embeddings or None if not available
        """
        if self.wn_embedder is None:
            logger.debug("WordNet-Affect embedder not available")
            return None
        
        embeddings = []
        for text in texts:
            try:
                # Get emotion embedding for text
                emb = self.wn_embedder.get_emotion_embedding(text)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    # Create zero embedding if no emotion words found
                    embeddings.append(np.zeros(300))  # Standard WN-Affect dimension
                    
            except Exception as e:
                logger.debug(f"WordNet embedding failed for text: {str(e)}")
                embeddings.append(np.zeros(300))
        
        return np.array(embeddings)
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        ids: List[str], 
        context_turns: Optional[int] = None,
        dialogue_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate Config146-style embeddings with dynamic K-turn context windows.
        
        Args:
            texts: List of text strings
            ids: List of utterance IDs
            context_turns: Number of context turns (None = use config default)
            dialogue_ids: List of dialogue IDs for boundary detection
            **kwargs: Additional parameters
            
        Returns:
            List of (embedding, id) tuples where embedding has shape (context_turns, embedding_dim)
        """
        if context_turns is None:
            context_turns = self.config['context_turns']
            
        logger.info(f"Generating Config146 embeddings for {len(texts)} texts with K={context_turns} context...")
        
        # Generate sentence embeddings
        logger.info("🔄 Generating sentence embeddings...")
        sentence_embeddings = self._get_sentence_embeddings(texts)
        logger.info(f"✅ Sentence embeddings shape: {sentence_embeddings.shape}")
        
        # Generate word embeddings if available
        word_embeddings = None
        if self.wn_embedder is not None:
            logger.info("🔄 Generating WordNet-Affect embeddings...")
            word_embeddings = self._get_word_embeddings(texts)
            if word_embeddings is not None:
                logger.info(f"✅ WordNet embeddings shape: {word_embeddings.shape}")
        
        # Combine embeddings using Config146 method (sum)
        logger.info("🔄 Combining embeddings using Config146 method...")
        combined_embeddings = []
        for i in range(len(texts)):
            sent_emb = sentence_embeddings[i]
            word_emb = word_embeddings[i] if word_embeddings is not None else None
            combined = self._combine_embeddings(sent_emb, word_emb)
            combined_embeddings.append(combined)
        
        logger.info(f"✅ Combined embeddings shape: {len(combined_embeddings)} x {combined_embeddings[0].shape}")
        
        # Create context windows with dialogue boundary awareness
        logger.info(f"🔄 Creating K={context_turns} context windows...")
        windowed_embeddings = self._create_context_window(
            combined_embeddings, 
            ids, 
            context_turns=context_turns,
            dialogue_ids=dialogue_ids
        )
        
        logger.info(f"✅ Generated {len(windowed_embeddings)} Config146 context windows")
        if len(windowed_embeddings) > 0:
            sample_shape = windowed_embeddings[0][0].shape
            logger.info(f"Sample embedding shape: {sample_shape}")
        
        return windowed_embeddings
    
    def generate_multiple_k_embeddings(
        self,
        texts: List[str],
        ids: List[str], 
        k_values: List[int],
        dialogue_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[int, List[Tuple[np.ndarray, str]]]:
        """
        Generate embeddings for multiple K values efficiently.
        
        This method generates embeddings once and creates different context windows,
        which is more efficient than calling generate_embeddings multiple times.
        
        Args:
            texts: List of text strings
            ids: List of utterance IDs
            k_values: List of K values to generate (e.g., [0, 2, 4, 6])
            dialogue_ids: List of dialogue IDs for boundary detection
            output_dir: Optional directory to save embeddings
            
        Returns:
            Dictionary mapping K values to embedding lists
        """
        logger.info(f"Generating Config146 embeddings for K values: {k_values}")
        
        # Generate base embeddings once
        logger.info("🔄 Generating base sentence embeddings...")
        sentence_embeddings = self._get_sentence_embeddings(texts)
        
        word_embeddings = None
        if self.wn_embedder is not None:
            logger.info("🔄 Generating base WordNet embeddings...")
            word_embeddings = self._get_word_embeddings(texts)
        
        # Combine embeddings once
        logger.info("🔄 Combining base embeddings...")
        combined_embeddings = []
        for i in range(len(texts)):
            sent_emb = sentence_embeddings[i]
            word_emb = word_embeddings[i] if word_embeddings is not None else None
            combined = self._combine_embeddings(sent_emb, word_emb)
            combined_embeddings.append(combined)
        
        # Generate different context windows for each K
        results = {}
        for k in k_values:
            logger.info(f"🔄 Creating K={k} context windows...")
            windowed_embeddings = self._create_context_window(
                combined_embeddings,
                ids,
                context_turns=k if k > 0 else 1,  # K=0 means just current utterance
                dialogue_ids=dialogue_ids
            )
            
            # For K=0, take only the current utterance (last in context window)
            if k == 0:
                windowed_embeddings = [
                    (emb[-1:], utterance_id) for emb, utterance_id in windowed_embeddings
                ]
            
            results[k] = windowed_embeddings
            logger.info(f"✅ Generated {len(windowed_embeddings)} embeddings for K={k}")
            
            # Save if output directory provided
            if output_dir:
                from pathlib import Path
                output_path = Path(output_dir) / f"config146_k{k}_embeddings.pkl"
                self.save_embeddings(
                    windowed_embeddings, 
                    output_path,
                    metadata={'k_value': k, 'method': 'config146'}
                )
        
        logger.info(f"✅ Completed multi-K embedding generation for {k_values}")
        return results