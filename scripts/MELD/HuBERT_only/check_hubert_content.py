import io
import numpy as np
from google.cloud import storage

def load_npz_from_gcs(bucket_name, blob_name):
    # 1. Initialize the GCS client (uses your GOOGLE_APPLICATION_CREDENTIALS)
    client = storage.Client(project="gen-lang-client-0105254213")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 2. Download the contents as bytes
    content = blob.download_as_bytes()

    # 3. Load into NumPy using a BytesIO buffer
    # Use 'with' to ensure the NpzFile object is closed correctly
    with np.load(io.BytesIO(content)) as data:
        # Access your features (e.g., data['features'])
        return {key: data[key] for key in data.files}

# Example Usage
# For path: gs://meld/hubert_features/dia0_utt0.npz
features = load_npz_from_gcs("meld", "hubert_features/meld_train_hubert.npz")
print(f"Loaded keys: {features.keys()}")
print(f"Features Shape: {features['features'].shape}")
print(f"Labels Shape: {features['labels'].shape}")
print(f"Sample Emotion Names: {features['emotion_names'][:7]}")

"""
Loaded keys: dict_keys(['features', 'labels', 'ids', 'emotion_names'])
Features Shape: (9988, 1024)
Labels Shape: (9988,)
Sample Emotion Names: ['anger' 'disgust' 'fear' 'joy' 'neutral' 'sadness' 'surprise']

"""