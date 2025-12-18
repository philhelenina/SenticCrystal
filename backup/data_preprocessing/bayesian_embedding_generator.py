"""
Bayesian Config146 Embedding Generator for SenticCrystal.

Extends the original Config146 with Bayesian Context LSTM for uncertainty quantification.
Combines WordNet-Affect + Sentence-RoBERTa with Bayesian 4-turn context modeling.
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

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
from models.simple_bayesian import SimpleBayesianLSTM, SimpleBayesianClassifier

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

class BayesianConfig146EmbeddingGenerator:
    """
    Bayesian Config146 Embedding Generator with uncertainty quantification.
    
    Configuration (same as original Config146):
    - apply_word_pe: False
    - pooling_method: "weighted_mean"
    - apply_sentence_pe: False  
    - combination_method: "sum"
    - bayesian_method: "context_lstm"  ← NOW ACTUALLY BAYESIAN!
    - context_turns: 4
    """
    
    def __init__(self, device: str = 'cpu', context_turns: int = 4, dropout: float = 0.3):
        self.device = device
        self.context_turns = context_turns
        self.dropout = dropout
        
        # Config146 settings
        self.config = {
            'apply_word_pe': False,
            'pooling_method': 'weighted_mean',
            'apply_sentence_pe': False,
            'combination_method': 'sum',
            'bayesian_method': 'context_lstm',
            'context_turns': context_turns
        }
        
        logger.info(f"🚀 Initializing Bayesian Config146 on {device}")
        logger.info(f"Context turns: {context_turns}, Dropout: {dropout}")
        logger.info(f"Config: {self.config}")
        
        # Initialize components
        self._init_sentence_embedder()
        self._init_word_embedder()
        
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
                corpus_path = LINGUA_EMOCA_DIR / 'src' / 'wn-affect-1.1' / 'wn-domains-3.2'
                
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
        """Get sentence embedding using original or fallback method."""
        try:
            if hasattr(self, 'sentence_embedder') and self.sentence_embedder is not None:
                # Use original SentenceEmbedder
                embedding = self.sentence_embedder.get_sentence_embedding(text)
                return np.array(embedding, dtype=np.float32)
            else:
                # Use fallback SentenceTransformer
                embedding = self.sentence_model.encode(text)
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting sentence embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(768, dtype=np.float32)
    
    def _get_word_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get WordNet-Affect embedding if available."""
        if self.word_embedder is None:
            return None
            
        try:
            word_embedding = self.word_embedder.get_word_embedding(text)
            if word_embedding is not None:
                return np.array(word_embedding, dtype=np.float32)
        except Exception as e:
            logger.debug(f"WordNet embedding failed: {e}")
        
        return None
    
    def _combine_embeddings(self, sent_emb: np.ndarray, word_emb: Optional[np.ndarray]) -> np.ndarray:
        """
        Combine sentence and word embeddings according to Config146.
        
        Config146 uses combination_method: "sum"
        """
        if word_emb is None:
            return sent_emb.astype(np.float32)
        
        # Ensure compatible shapes
        if sent_emb.shape[0] != word_emb.shape[0]:
            # Pad or truncate to match
            min_dim = min(sent_emb.shape[0], word_emb.shape[0])
            sent_emb = sent_emb[:min_dim]
            word_emb = word_emb[:min_dim]
        
        # Config146: sum combination
        combined = sent_emb + word_emb
        return combined.astype(np.float32)
    
    def _apply_pooling(self, context_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Apply weighted_mean pooling according to Config146.
        
        Config146 uses pooling_method: "weighted_mean" - more recent contexts get higher weights.
        """
        if len(context_embeddings) == 1:
            return context_embeddings[0]
        
        # Weight more recent contexts higher (Config146 style)
        n = len(context_embeddings)
        weights = np.array([(i + 1) / n for i in range(n)])  # [1/n, 2/n, ..., n/n]
        weights = weights / weights.sum()  # Normalize
        
        # Apply weighted average
        weighted_emb = np.average(context_embeddings, axis=0, weights=weights)
        return weighted_emb.astype(np.float32)
    
    def generate_embeddings(self, texts: List[str], ids: List[str]) -> List[Tuple[np.ndarray, str]]:
        """
        Generate Bayesian Config146 embeddings with context modeling.
        
        Returns embeddings that can be used with BayesianContextLSTM.
        """
        logger.info(f"🚀 Generating Bayesian Config146 embeddings for {len(texts)} texts")
        logger.info(f"Context turns: {self.context_turns}")
        
        results = []
        
        for i, (text, text_id) in enumerate(zip(texts, ids)):
            # Get context window (Config146: 4 turns)
            start_idx = max(0, i - self.context_turns + 1)
            context_texts = texts[start_idx:i+1]
            
            # Generate embeddings for context
            context_embeddings = []
            for ctx_text in context_texts:
                # Get sentence embedding
                sent_emb = self._get_sentence_embedding(ctx_text)
                
                # Get word embedding (WordNet-Affect)
                word_emb = self._get_word_embedding(ctx_text)
                
                # Combine embeddings (Config146: sum)
                combined_emb = self._combine_embeddings(sent_emb, word_emb)
                context_embeddings.append(combined_emb)
            
            # Apply pooling (Config146: weighted_mean)
            final_embedding = self._apply_pooling(context_embeddings)
            
            # Ensure fixed dimensionality for LSTM
            if final_embedding.shape[0] != 768:
                # Pad or truncate to 768 dimensions
                if final_embedding.shape[0] < 768:
                    padding = np.zeros(768 - final_embedding.shape[0], dtype=np.float32)
                    final_embedding = np.concatenate([final_embedding, padding])
                else:
                    final_embedding = final_embedding[:768]
            
            results.append((final_embedding, text_id))
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(texts)} texts")
        
        logger.info(f"✅ Generated {len(results)} Bayesian Config146 embeddings")
        return results

class BayesianConfig146Classifier(nn.Module):
    """
    Complete Bayesian Config146 classifier with uncertainty quantification.
    
    Combines the embedding generation with Bayesian Context LSTM classification.
    """
    
    def __init__(self, num_classes: int = 4, hidden_size: int = 256, dropout: float = 0.3, device: str = 'cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        # Bayesian Context LSTM (Config146 style)
        self.bayesian_context_lstm = SimpleBayesianLSTM(
            input_size=768,  # SentenceTransformer + WordNet-Affect
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        # Classification head with uncertainty
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.to(device)
        logger.info(f"✅ BayesianConfig146Classifier: {hidden_size}→{num_classes}, device={device}")
    
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Bayesian uncertainty quantification.
        
        Args:
            x: Input embeddings (batch_size, seq_len, 768) or (batch_size, 768)
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with predictions, probabilities, uncertainty, and confidence
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # (batch_size, 768) → (batch_size, 1, 768) for LSTM
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Bayesian Context LSTM forward
        lstm_out, lstm_uncertainty = self.bayesian_context_lstm(x, n_samples=n_samples)
        
        # Classification with sampling
        classification_samples = []
        for _ in range(n_samples):
            self.train()  # Enable dropout for sampling
            logits = self.classifier(lstm_out)
            probs = torch.softmax(logits, dim=-1)
            classification_samples.append(probs)
        
        # Calculate statistics
        classification_samples = torch.stack(classification_samples)  # (n_samples, batch_size, num_classes)
        mean_probs = torch.mean(classification_samples, dim=0)
        classification_uncertainty = torch.var(classification_samples, dim=0)
        
        # Predictions and confidence
        max_prob, predicted_class = torch.max(mean_probs, dim=-1)
        total_uncertainty = torch.sum(classification_uncertainty, dim=-1)
        confidence = max_prob / (1.0 + total_uncertainty)  # Normalized confidence
        
        return {
            'predictions': predicted_class,
            'probabilities': mean_probs,
            'max_probability': max_prob,
            'classification_uncertainty': total_uncertainty,
            'lstm_uncertainty': lstm_uncertainty,
            'confidence': confidence,
            'needs_review': confidence < 0.7  # Threshold for human review
        }
    
    def predict_with_uncertainty(self, embeddings: List[Tuple[np.ndarray, str]], 
                                n_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Predict with uncertainty for a batch of embeddings.
        
        Args:
            embeddings: List of (embedding, text_id) tuples from BayesianConfig146EmbeddingGenerator
            n_samples: Number of Monte Carlo samples
            
        Returns:
            List of prediction dictionaries with uncertainty information
        """
        self.eval()
        
        # Convert embeddings to tensor
        embedding_tensors = []
        text_ids = []
        
        for emb, text_id in embeddings:
            embedding_tensors.append(torch.FloatTensor(emb))
            text_ids.append(text_id)
        
        # Stack into batch
        batch_embeddings = torch.stack(embedding_tensors).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            results = self.forward(batch_embeddings, n_samples=n_samples)
        
        # Format results
        predictions = []
        for i in range(len(text_ids)):
            pred_dict = {
                'text_id': text_ids[i],
                'predicted_class': results['predictions'][i].item(),
                'probabilities': results['probabilities'][i].cpu().numpy().tolist(),
                'max_probability': results['max_probability'][i].item(),
                'confidence': results['confidence'][i].item(),
                'uncertainty': results['classification_uncertainty'][i].item(),
                'needs_review': results['needs_review'][i].item(),
                'lstm_uncertainty': results['lstm_uncertainty'][i].mean().item() if results['lstm_uncertainty'] is not None else 0.0
            }
            predictions.append(pred_dict)
        
        return predictions