"""
GCP Storage data loader for SenticCrystal.

Downloads and caches data from Google Cloud Storage buckets.
Optimized for MacBook Air M4.
"""

import os
from pathlib import Path
import logging
from typing import Optional, List
import hashlib
import json

try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logging.warning("Google Cloud Storage not available. Install with: pip install google-cloud-storage")

logger = logging.getLogger(__name__)

class GCPDataLoader:
    """
    Google Cloud Storage data loader with local caching.
    """
    
    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        cache_dir: str = "./data_cache"
    ):
        """
        Initialize GCP data loader.
        
        Args:
            bucket_name: GCP bucket name
            project_id: GCP project ID (optional)
            credentials_path: Path to service account JSON (optional)
            cache_dir: Local cache directory
        """
        if not GCP_AVAILABLE:
            raise ImportError("Google Cloud Storage not available. Install with: pip install google-cloud-storage")
        
        self.bucket_name = bucket_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup credentials
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Initialize client
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Connected to GCP bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to connect to GCP: {e}")
            raise
        
        # Cache metadata
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self) -> dict:
        """Load cache metadata from local file."""
        if self.cache_metadata_file.exists():
            with open(self.cache_metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to local file."""
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(self.cache_metadata, f, indent=2)
    
    def _get_blob_hash(self, blob_name: str) -> str:
        """Get hash of blob for cache validation."""
        try:
            blob = self.bucket.blob(blob_name)
            blob.reload()
            # Use etag as hash (GCP's content hash)
            return blob.etag
        except Exception as e:
            logger.warning(f"Could not get hash for {blob_name}: {e}")
            return ""
    
    def _is_cached_and_valid(self, blob_name: str, local_path: Path) -> bool:
        """Check if file is cached and valid."""
        if not local_path.exists():
            return False
        
        # Check metadata
        if blob_name in self.cache_metadata:
            cached_hash = self.cache_metadata[blob_name].get('hash', '')
            current_hash = self._get_blob_hash(blob_name)
            
            if cached_hash == current_hash and current_hash:
                logger.debug(f"Cache hit: {blob_name}")
                return True
        
        logger.debug(f"Cache miss: {blob_name}")
        return False
    
    def download_file(
        self, 
        blob_name: str, 
        local_path: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download file from GCP bucket with caching.
        
        Args:
            blob_name: Name of blob in bucket
            local_path: Local path to save file (optional)
            force_download: Force download even if cached
            
        Returns:
            Path to local file
        """
        # Determine local path
        if local_path:
            local_file = Path(local_path)
        else:
            local_file = self.cache_dir / blob_name
        
        # Create directory if needed
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        if not force_download and self._is_cached_and_valid(blob_name, local_file):
            logger.info(f"Using cached file: {local_file}")
            return local_file
        
        # Download from GCP
        try:
            logger.info(f"Downloading {blob_name} from GCP bucket...")
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(str(local_file))
            
            # Update cache metadata
            blob_hash = self._get_blob_hash(blob_name)
            self.cache_metadata[blob_name] = {
                'hash': blob_hash,
                'local_path': str(local_file),
                'size': local_file.stat().st_size if local_file.exists() else 0
            }
            self._save_cache_metadata()
            
            logger.info(f"Downloaded: {local_file}")
            return local_file
            
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            raise
    
    def download_directory(
        self, 
        prefix: str, 
        local_dir: Optional[str] = None,
        force_download: bool = False
    ) -> List[Path]:
        """
        Download all files with given prefix (directory-like).
        
        Args:
            prefix: Blob name prefix (directory path)
            local_dir: Local directory to save files
            force_download: Force download even if cached
            
        Returns:
            List of local file paths
        """
        if local_dir:
            local_path = Path(local_dir)
        else:
            local_path = self.cache_dir / prefix
        
        local_path.mkdir(parents=True, exist_ok=True)
        
        # List blobs with prefix
        blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
        logger.info(f"Found {len(blobs)} files with prefix '{prefix}'")
        
        downloaded_files = []
        for blob in blobs:
            # Skip directories
            if blob.name.endswith('/'):
                continue
            
            # Calculate relative path
            relative_path = blob.name[len(prefix):].lstrip('/')
            local_file_path = local_path / relative_path
            
            # Download file
            downloaded_file = self.download_file(
                blob.name, 
                str(local_file_path),
                force_download
            )
            downloaded_files.append(downloaded_file)
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {local_path}")
        return downloaded_files
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List all files in bucket with given prefix.
        
        Args:
            prefix: Blob name prefix to filter by
            
        Returns:
            List of blob names
        """
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        file_names = [blob.name for blob in blobs if not blob.name.endswith('/')]
        return file_names
    
    def get_cache_info(self) -> dict:
        """Get information about cached files."""
        cache_info = {
            'cache_dir': str(self.cache_dir),
            'total_files': len(self.cache_metadata),
            'total_size_mb': sum(
                meta.get('size', 0) for meta in self.cache_metadata.values()
            ) / (1024 * 1024),
            'files': self.cache_metadata
        }
        return cache_info
    
    def clear_cache(self, confirm: bool = False):
        """
        Clear local cache.
        
        Args:
            confirm: Must be True to actually clear cache
        """
        if not confirm:
            logger.warning("Cache not cleared. Set confirm=True to clear cache.")
            return
        
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_metadata = {}
            self._save_cache_metadata()
            logger.info("Cache cleared")

def setup_gcp_data_loader(
    bucket_name: str,
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    cache_dir: str = "./data_cache"
) -> GCPDataLoader:
    """
    Convenience function to setup GCP data loader.
    
    Args:
        bucket_name: GCP bucket name
        project_id: GCP project ID
        credentials_path: Path to service account JSON
        cache_dir: Local cache directory
        
    Returns:
        GCPDataLoader instance
    """
    return GCPDataLoader(
        bucket_name=bucket_name,
        project_id=project_id,
        credentials_path=credentials_path,
        cache_dir=cache_dir
    )

# Example usage functions
def download_iemocap_from_gcp(
    bucket_name: str,
    iemocap_prefix: str = "datasets/iemocap/",
    local_dir: str = "./data/iemocap",
    **gcp_kwargs
) -> List[Path]:
    """Download IEMOCAP data from GCP bucket."""
    loader = setup_gcp_data_loader(bucket_name, **gcp_kwargs)
    return loader.download_directory(iemocap_prefix, local_dir)

def download_meld_from_gcp(
    bucket_name: str,
    meld_prefix: str = "datasets/meld/",
    local_dir: str = "./data/meld",
    **gcp_kwargs
) -> List[Path]:
    """Download MELD data from GCP bucket."""
    loader = setup_gcp_data_loader(bucket_name, **gcp_kwargs)
    return loader.download_directory(meld_prefix, local_dir)