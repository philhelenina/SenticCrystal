import os
import sys
import pkgutil

# 1. Setup local_lib pathing
local_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../local_lib"))

if local_lib_path not in sys.path:
    sys.path.insert(0, local_lib_path)

# Handle Google Namespace issue
if os.path.exists(os.path.join(local_lib_path, "google")):
    import google
    google.__path__ = pkgutil.extend_path(google.__path__, google.__name__)

# Now we can safely import these
from dotenv import load_dotenv
from google.cloud import storage

# 2. Load Environment Variables
# Finds .env in the root (up two levels from scripts/MELD/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, "..", "..", ".env")
load_dotenv(env_path)

def list_meld_files():
    # 3. Get the key path from .env
    # Change "MELD_READ_KEY" below if you named it differently in your .env
    key_path = os.getenv("MELD_READ_KEY") 

    if not key_path:
        print("Error: MELD_READ_KEY not found in .env file.")
        return
    
    if not os.path.exists(key_path):
        print(f"Error: Physical JSON key not found at {key_path}")
        return

    try:
        # 4. Initialize Client with the specific key
        client = storage.Client.from_service_account_json(key_path)
        
        buckets = list(client.list_buckets())
        print(f"Total buckets found: {len(buckets)}")
        
        for bucket in buckets:
            print(f"\nScanning Bucket: {bucket.name}")
            # List first 5 files to verify access
            blobs = client.list_blobs(bucket.name, max_results=5)
            for blob in blobs:
                print(f" - Found: {blob.name}")
                
    except Exception as e:
        print(f"GCP Connection Error: {e}")

if __name__ == "__main__":
    list_meld_files()
