import opensmile
import pandas as pd
import io
from google.cloud import storage
from tqdm import tqdm
import soundfile as sf
import numpy as np
import tempfile
import subprocess
from pathlib import Path
import os


# does the actual extraction of audio features and maps to dialogue id

# configuring/initializing the connection to GCP and opensmile setup

client = storage.Client(project="gen-lang-client-0105254213")
bucket = client.bucket("meld")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def process_blob(blob):
    audio_bytes = blob.download_as_bytes(timeout=30)
    
    # tmp files for media conversion
    tmp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    try:
        # converting each file to a suitable format,
        # input convert to 16kHz mono audio and save as tmp wav/mp4 files file
        tmp_mp4.write(audio_bytes)
        tmp_mp4.flush()
        tmp_mp4.close()
        tmp_wav.close()

        # using ffmpeg to transform the audio file so openSMILE can analyze
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp4.name,
             "-ar", "16000", "-ac", "1", tmp_wav.name],
            capture_output=True, timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        
        audio, sr = sf.read(tmp_wav.name)
    finally:
        os.unlink(tmp_mp4.name)
        os.unlink(tmp_wav.name)
    # extraction and conversion to a dictionary to be appended to csv later
    feats = smile.process_signal(audio, sr)
    row = feats.iloc[0].to_dict()
    row["filename"] = Path(blob.name).stem
    return row

splits = {
    "train": "train/",
    "dev":   "dev_splits_complete/",
    "test":  "output_repeated_splits_test/",
}

# Checkpoint — skip already done files
done = set()
csv_path = "/home/liaojd/SenticCrystal/scripts/MELD/opensmile/meld_egemaps_raw.csv"
if os.path.exists(csv_path):
    done = set(pd.read_csv(csv_path)["filename"].tolist())
    print(f"Skipping {len(done)} already processed files")

records = []
for split_name, prefix in splits.items():
    # lists files in specific folder
    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in tqdm(blobs, desc=split_name):
        if blob.name.endswith((".wav", ".mp4")):
            if Path(blob.name).stem in done:  # skip if already done
                continue
            try:
                row = process_blob(blob)
                row["split"] = split_name
                records.append(row)
            except (Exception, subprocess.TimeoutExpired) as e:
                print(f"✗ {blob.name}: {e}")

# Append to existing CSV rather than overwrite
df = pd.DataFrame(records)
if os.path.exists(csv_path):
    df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
df.to_csv(csv_path, index=False)
print(f"Done → {len(df)} rows × {df.shape[1]} cols")