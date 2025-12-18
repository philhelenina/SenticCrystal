"""
Saturn Cloud A100 Device Utilities
==================================

Optimized device management for NVIDIA A100 GPUs on Saturn Cloud.
"""

import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_torch_for_a100():
    """
    Configure PyTorch for optimal performance on Saturn Cloud A100 GPUs.
    
    Returns:
        torch.device: The optimal device for computation
    """
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available! Falling back to CPU")
        return torch.device('cpu')
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    logger.info(f"🎯 Found {gpu_count} CUDA device(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Use first GPU (Saturn Cloud typically provides A100)
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    # A100-specific optimizations
    logger.info("🔧 Applying A100 optimizations...")
    
    # Enable TensorFloat-32 (TF32) for A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("✅ TF32 enabled for faster training")
    
    # Optimize memory allocation
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    logger.info("✅ CUDA memory allocation optimized")
    
    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    logger.info("✅ cuDNN benchmarking enabled")
    
    # Set optimal number of threads
    if torch.get_num_threads() < 8:
        torch.set_num_threads(8)
    logger.info(f"✅ PyTorch threads: {torch.get_num_threads()}")
    
    # Memory info
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    logger.info(f"📊 GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    logger.info(f"🚀 Ready for training on {device}")
    return device

def get_optimal_batch_size(model_size='medium'):
    """
    Get optimal batch size for A100 based on model complexity.
    
    Args:
        model_size (str): 'small', 'medium', 'large'
    
    Returns:
        int: Recommended batch size
    """
    
    if not torch.cuda.is_available():
        return 16  # Conservative for CPU
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory >= 80:  # A100 80GB
        batch_sizes = {'small': 128, 'medium': 64, 'large': 32}
    elif gpu_memory >= 40:  # A100 40GB
        batch_sizes = {'small': 64, 'medium': 32, 'large': 16}
    else:
        batch_sizes = {'small': 32, 'medium': 16, 'large': 8}
    
    recommended = batch_sizes.get(model_size, 32)
    logger.info(f"💡 Recommended batch size for {model_size} model: {recommended}")
    return recommended

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 GPU memory cache cleared")

def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    return 0, 0

def log_memory_usage(stage=""):
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated, reserved = get_memory_usage()
        logger.info(f"📊 {stage} GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Test function
if __name__ == "__main__":
    print("🧪 Testing Saturn Cloud A100 setup...")
    device = setup_torch_for_a100()
    
    print(f"\n📋 Device Information:")
    print(f"  Device: {device}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test tensor operations
        print(f"\n🧪 Testing tensor operations...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.mm(x, x.T)
        print(f"✅ Matrix multiplication successful: {y.shape}")
        
        log_memory_usage("After test")
    
    print(f"\n✅ Saturn Cloud A100 setup verification complete!")