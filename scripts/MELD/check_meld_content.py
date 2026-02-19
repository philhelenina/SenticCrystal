import os
import sys
import pkgutil

local_lib_path = os.path.abspath("local_lib")

# 2. Add it to the path
if local_lib_path not in sys.path:
    sys.path.insert(0, local_lib_path)

if os.path.exists(os.path.join(local_lib_path, "google")):
    import google
    google.__path__ = pkgutil.extend_path(google.__path__, google.__name__)

from google.cloud import storage
# Use an absolute path to find your key in the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If your script is in /scripts/, go up one level to find the key in root
PATH_TO_KEY = os.path.join(BASE_DIR, "..", "gen-lang-client-0105254213-eda60051d891.json")

def list_meld_files():
    # Use the absolute path here
    if not os.path.exists(PATH_TO_KEY):
        print(f"Error: Key not found at {PATH_TO_KEY}")
        return

    client = storage.Client.from_service_account_json(PATH_TO_KEY)
    
    buckets = list(client.list_buckets())
    print(f"Total buckets found: {len(buckets)}")
    for bucket in buckets:
        print(f"\nScanning Bucket: {bucket.name}")
        blobs = client.list_blobs(bucket.name, max_results=5)
        for blob in blobs:
            print(f" - Found: {blob.name}")

if __name__ == "__main__":
    list_meld_files()