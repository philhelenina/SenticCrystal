"""
Bayesian Config146 Embedding Generator for SenticCrystal.

Refactored implementation extending Config146Generator with Bayesian uncertainty quantification.
Combines Config146 architecture with Bayesian Context LSTM for uncertainty estimation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import logging

# Import base and Config146 classes
from .config146_generator import Config146EmbeddingGenerator

# Import Bayesian modules
sys.path.append(str(Path(__file__).parent.parent))
try:
    from models.simple_bayesian import SimpleBayesianLSTM, SimpleBayesianClassifier
    from models.bayesian_modules import BayesianContextLSTM
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Bayesian modules not found: {e}")
    SimpleBayesianLSTM = None
    SimpleBayesianClassifier = None
    BayesianContextLSTM = None

logger = logging.getLogger(__name__)


class BayesianConfig146EmbeddingGenerator(Config146EmbeddingGenerator):
    """
    Bayesian Config146 Embedding Generator with uncertainty quantification.
    
    Extends Config146Generator with:
    - Bayesian Context LSTM for true Bayesian inference
    - Monte Carlo Dropout for uncertainty estimation  
    - Confidence scoring and human review flagging
    - Uncertainty-aware embedding generation
    
    Configuration (same as Config146 + Bayesian):
    - All Config146 settings
    - bayesian_method: "context_lstm" (now actually Bayesian!)
    - dropout: 0.3 (for MC Dropout)
    - mc_samples: 10 (Monte Carlo samples)
    """
    
    def __init__(
        self, 
        device: str = 'cpu', 
        config: Optional[Dict[str, Any]] = None,
        dropout: float = 0.3,
        mc_samples: int = 10
    ):
        """
        Initialize Bayesian Config146 embedding generator.
        
        Args:
            device: Computing device ('cpu', 'cuda', 'mps')
            config: Configuration dictionary (extends Config146 defaults)
            dropout: Dropout rate for MC Dropout
            mc_samples: Number of Monte Carlo samples for uncertainty
        """
        # Extend config with Bayesian settings
        bayesian_config = config or {}
        bayesian_config.update({
            'dropout': dropout,
            'mc_samples': mc_samples,
            'bayesian_method': 'context_lstm',  # Force Bayesian method
            'uncertainty_estimation': True
        })
        
        super().__init__(device=device, config=bayesian_config)
        
        # Initialize Bayesian-specific components
        self.dropout = dropout
        self.mc_samples = mc_samples
        self.bayesian_lstm = None
        
        self._initialize_bayesian_components()
    
    def _initialize_bayesian_components(self):
        """Initialize Bayesian-specific components."""
        try:
            logger.info("🧠 Initializing Bayesian components...")
            
            # Initialize Bayesian Context LSTM if available
            if BayesianContextLSTM is not None:
                embedding_dim = self.config['embedding_dim']
                context_turns = self.config['context_turns']
                self.bayesian_lstm = BayesianContextLSTM(
                    input_size=embedding_dim,
                    hidden_size=embedding_dim//2,  # Reduce to prevent memory issues
                    num_layers=2,
                    context_turns=context_turns
                ).to(self.device)
                
                logger.info(f"✅ Initialized BayesianContextLSTM (dim: {embedding_dim})")
                
            else:
                logger.warning("⚠️ BayesianContextLSTM not available, using MC Dropout fallback")
            
            # Log Bayesian settings
            logger.info("🔧 Bayesian Settings:")
            logger.info(f"  dropout: {self.dropout}")
            logger.info(f"  mc_samples: {self.mc_samples}")
            logger.info(f"  bayesian_method: {self.config['bayesian_method']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bayesian components: {str(e)}")
            raise
    
    def _apply_bayesian_processing(
        self, 
        embeddings: List[np.ndarray], 
        context_turns: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Apply Bayesian processing to embeddings for uncertainty quantification.
        
        Args:
            embeddings: List of combined embeddings
            context_turns: Number of context turns
            
        Returns:
            Tuple of (processed_embeddings, uncertainty_scores)
        """
        if self.bayesian_lstm is None:
            # Fallback: return original embeddings with dummy uncertainties
            logger.debug("No Bayesian LSTM available, using original embeddings")
            uncertainties = [0.0] * len(embeddings)
            return embeddings, uncertainties
        
        logger.info("🧠 Applying Bayesian processing...")
        
        processed_embeddings = []
        uncertainty_scores = []
        
        # Convert to tensors
        embedding_tensors = [torch.FloatTensor(emb).unsqueeze(0).to(self.device) 
                           for emb in embeddings]
        
        # Process through Bayesian LSTM with MC samples
        self.bayesian_lstm.train()  # Enable dropout for MC sampling
        
        for emb_tensor in embedding_tensors:
            mc_outputs = []
            
            # Generate MC samples
            for _ in range(self.mc_samples):
                with torch.no_grad():
                    output, _ = self.bayesian_lstm(emb_tensor)
                    mc_outputs.append(output.squeeze(0).cpu().numpy())
            
            # Calculate mean and uncertainty
            mc_outputs = np.array(mc_outputs)  # (mc_samples, embedding_dim)
            mean_output = np.mean(mc_outputs, axis=0)
            uncertainty = np.std(mc_outputs, axis=0).mean()  # Average std as uncertainty
            
            processed_embeddings.append(mean_output)
            uncertainty_scores.append(float(uncertainty))
        
        logger.info(f"✅ Bayesian processing completed, avg uncertainty: {np.mean(uncertainty_scores):.4f}")
        return processed_embeddings, uncertainty_scores
    
    def _calculate_confidence_scores(self, uncertainty_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Calculate confidence scores and human review flags based on uncertainty.
        
        Args:
            uncertainty_scores: List of uncertainty values
            
        Returns:
            List of confidence dictionaries
        """
        confidence_info = []
        
        # Calculate uncertainty statistics
        uncertainties = np.array(uncertainty_scores)
        mean_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)
        high_uncertainty_threshold = mean_uncertainty + 2 * std_uncertainty
        
        logger.info(f"📊 Uncertainty statistics - Mean: {mean_uncertainty:.4f}, Std: {std_uncertainty:.4f}")
        
        for i, uncertainty in enumerate(uncertainty_scores):
            # Convert uncertainty to confidence (inverse relationship)
            confidence = 1.0 / (1.0 + uncertainty)
            
            # Flag for human review if high uncertainty
            needs_review = uncertainty > high_uncertainty_threshold
            
            # Categorize confidence level
            if confidence > 0.8:
                confidence_level = "high"
            elif confidence > 0.6:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            confidence_info.append({
                'confidence_score': float(confidence),
                'uncertainty_score': float(uncertainty),
                'confidence_level': confidence_level,
                'needs_human_review': needs_review,
                'utterance_index': i
            })
        
        review_count = sum(1 for info in confidence_info if info['needs_human_review'])
        logger.info(f"🚨 {review_count}/{len(confidence_info)} utterances flagged for human review")
        
        return confidence_info
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        ids: List[str], 
        context_turns: Optional[int] = None,
        dialogue_ids: Optional[List[str]] = None,
        return_uncertainty: bool = True,
        **kwargs
    ) -> Tuple[List[Tuple[np.ndarray, str]], Optional[List[Dict[str, Any]]]]:
        """
        Generate Bayesian Config146 embeddings with uncertainty quantification.
        
        Args:
            texts: List of text strings
            ids: List of utterance IDs
            context_turns: Number of context turns (None = use config default)
            dialogue_ids: List of dialogue IDs for boundary detection
            return_uncertainty: Whether to return uncertainty information
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (embeddings, uncertainty_info) where uncertainty_info is None if return_uncertainty=False
        """
        if context_turns is None:
            context_turns = self.config['context_turns']
            
        logger.info(f"Generating Bayesian Config146 embeddings for {len(texts)} texts with K={context_turns}...")
        
        # Generate base embeddings using parent method
        logger.info("🔄 Generating base Config146 embeddings...")
        sentence_embeddings = self._get_sentence_embeddings(texts)
        
        word_embeddings = None
        if self.wn_embedder is not None:
            logger.info("🔄 Generating WordNet-Affect embeddings...")
            word_embeddings = self._get_word_embeddings(texts)
        
        # Combine embeddings
        logger.info("🔄 Combining embeddings...")
        combined_embeddings = []
        for i in range(len(texts)):
            sent_emb = sentence_embeddings[i]
            word_emb = word_embeddings[i] if word_embeddings is not None else None
            combined = self._combine_embeddings(sent_emb, word_emb)
            combined_embeddings.append(combined)
        
        # Apply Bayesian processing
        uncertainty_info = None
        if return_uncertainty:
            logger.info("🧠 Applying Bayesian uncertainty quantification...")
            bayesian_embeddings, uncertainty_scores = self._apply_bayesian_processing(
                combined_embeddings, context_turns
            )
            
            # Calculate confidence scores
            uncertainty_info = self._calculate_confidence_scores(uncertainty_scores)
            combined_embeddings = bayesian_embeddings
        
        # Create context windows
        logger.info(f"🔄 Creating K={context_turns} context windows...")
        windowed_embeddings = self._create_context_window(
            combined_embeddings, 
            ids, 
            context_turns=context_turns,
            dialogue_ids=dialogue_ids
        )
        
        logger.info(f"✅ Generated {len(windowed_embeddings)} Bayesian Config146 embeddings")
        if len(windowed_embeddings) > 0:
            sample_shape = windowed_embeddings[0][0].shape
            logger.info(f"Sample embedding shape: {sample_shape}")
        
        return windowed_embeddings, uncertainty_info
    
    def generate_with_confidence_filtering(
        self,
        texts: List[str],
        ids: List[str],
        confidence_threshold: float = 0.7,
        context_turns: Optional[int] = None,
        dialogue_ids: Optional[List[str]] = None
    ) -> Tuple[List[Tuple[np.ndarray, str]], List[str], List[Dict[str, Any]]]:
        """
        Generate embeddings and filter by confidence threshold.
        
        Args:
            texts: List of text strings
            ids: List of utterance IDs
            confidence_threshold: Minimum confidence score to include
            context_turns: Number of context turns
            dialogue_ids: List of dialogue IDs
            
        Returns:
            Tuple of (high_confidence_embeddings, low_confidence_ids, all_uncertainty_info)
        """
        logger.info(f"Generating embeddings with confidence filtering (threshold: {confidence_threshold})")
        
        # Generate embeddings with uncertainty
        embeddings, uncertainty_info = self.generate_embeddings(
            texts, ids, context_turns, dialogue_ids, return_uncertainty=True
        )
        
        # Filter by confidence
        high_confidence_embeddings = []
        low_confidence_ids = []
        
        for (emb, emb_id), uncertainty in zip(embeddings, uncertainty_info):
            if uncertainty['confidence_score'] >= confidence_threshold:
                high_confidence_embeddings.append((emb, emb_id))
            else:
                low_confidence_ids.append(emb_id)
        
        logger.info(f"✅ Confidence filtering results:")
        logger.info(f"  High confidence: {len(high_confidence_embeddings)}")
        logger.info(f"  Low confidence: {len(low_confidence_ids)}")
        logger.info(f"  Retention rate: {len(high_confidence_embeddings)/len(embeddings)*100:.1f}%")
        
        return high_confidence_embeddings, low_confidence_ids, uncertainty_info