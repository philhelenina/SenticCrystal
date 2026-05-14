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

# splits = {
#     "train": "train/",
#     "dev":   "dev_splits_complete/",
#     "test":  "output_repeated_splits_test/",
# }

# # Checkpoint — skip already done files
# done = set()
# csv_path = "/home/liaojd/SenticCrystal/scripts/MELD/opensmile/meld_egemaps_raw.csv"
# if os.path.exists(csv_path):
#     done = set(pd.read_csv(csv_path)["filename"].tolist())
#     print(f"Skipping {len(done)} already processed files")

# records = []
# for split_name, prefix in splits.items():
#     # lists files in specific folder
#     blobs = list(bucket.list_blobs(prefix=prefix))
#     for blob in tqdm(blobs, desc=split_name):
#         if blob.name.endswith((".wav", ".mp4")):
#             if Path(blob.name).stem in done:  # skip if already done
#                 continue
#             try:
#                 row = process_blob(blob)
#                 row["split"] = split_name
#                 records.append(row)
#             except (Exception, subprocess.TimeoutExpired) as e:
#                 print(f"✗ {blob.name}: {e}")

# # Append to existing CSV rather than overwrite
# df = pd.DataFrame(records)
# if os.path.exists(csv_path):
#     df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
# df.to_csv(csv_path, index=False)
# print(f"Done → {len(df)} rows × {df.shape[1]} cols")

csv_path = "/home/liaojd/SenticCrystal/scripts/MELD/opensmile/meld_egemaps_raw.csv"
df = pd.read_csv(csv_path)

# 2. Ensure Dialogue_ID exists (extracting from filename if necessary)
if 'Dialogue_ID' not in df.columns:
    df['Dialogue_ID'] = df['filename'].str.extract(r'dia(\d+)').astype(int)

# 3. Identify the columns to average
# We want to keep: split, Dialogue_ID, and Speaker
# we want to average: all the openSMILE features
exclude = ['filename', 'split', 'Dialogue_ID', 'Speaker', 'Emotion', 'Sentiment']
feature_cols = [c for c in df.columns if c not in exclude]

# 4. AGGREGATE
# This groups by Dialogue and Speaker and calculates the mean for all features
df_aggregated = df.groupby(['split', 'Dialogue_ID', 'Speaker'])[feature_cols].mean().reset_index()

# 5. Alphabetize Features (Per your project requirement)
metadata_cols = ['split', 'Dialogue_ID', 'Speaker']
sorted_features = sorted(feature_cols)
df_final = df_aggregated[metadata_cols + sorted_features]

# 6. Save
output_path = "/home/liaojd/SenticCrystal/scripts/MELD/opensmile/meld_egemaps_aggregated.csv"
df_final.to_csv(output_path, index=False)

print(f"Aggregation complete! Created {len(df_final)} rows.")
print(f"Columns are sorted alphabetically: {df_final.columns[3:6]} ...")
