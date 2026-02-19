#!/usr/bin/env python3
"""
preprocess_dailydialog.py - Preprocess DailyDialog to utterance-level format
============================================================================

Converts DailyDialog from dialogue-level to utterance-level format,
matching MELD format for unified training.

DailyDialog Native Labels:
- Dialogue Act: 1=inform, 2=question, 3=directive, 4=commissive
- Emotion: 0=neutral, 1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise

Output format (matching MELD):
- file_id: dialogue ID
- utterance_id: utterance index within dialogue
- utterance: text
- label: emotion label (string)
- speaker: A or B (alternating)
- dialogue_act: DA label (0-3, remapped)

Usage:
    python scripts/generator/preprocess_dailydialog.py
"""

import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "daily_raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dailydialog_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion mapping (DailyDialog -> string labels)
DD_EMOTION_MAP = {
    0: 'neutral',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',  # will map to 'joy' for consistency
    5: 'sadness',
    6: 'surprise'
}

# Map to match MELD labels
EMOTION_TO_MELD = {
    'neutral': 'neutral',
    'anger': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'happiness': 'joy',
    'sadness': 'sadness',
    'surprise': 'surprise'
}

# DA mapping (DailyDialog uses 1-4, we use 0-3)
DD_DA_MAP = {
    0: 0,  # unknown -> inform
    1: 0,  # inform -> inform
    2: 1,  # question -> question
    3: 2,  # directive -> directive
    4: 3   # commissive -> commissive
}

DA_NAMES = ['inform', 'question', 'directive', 'commissive']


def parse_dialog_string(dialog_str: str) -> list:
    """Parse the dialog string into a list of utterances.

    The format is like a numpy array:
    "['First utterance'
     'Second utterance'
     'Third utterance']"
    """
    # Remove leading/trailing brackets and whitespace
    clean = dialog_str.strip()
    if clean.startswith('['):
        clean = clean[1:]
    if clean.endswith(']'):
        clean = clean[:-1]

    # Split by quote boundaries - find all quoted strings
    # Match both single and double quotes, handling escaped quotes
    utterances = []

    # Pattern: match content between quotes (single or double)
    # Handle both ' and " delimiters
    pattern = r"'([^'\\]*(?:\\.[^'\\]*)*)'\s*|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"\s*"
    matches = re.findall(pattern, clean)

    for match in matches:
        # match is a tuple (single_quote_content, double_quote_content)
        text = match[0] if match[0] else match[1]
        text = text.strip()
        if text:
            # Unescape any escaped quotes
            text = text.replace("\\'", "'").replace('\\"', '"')
            utterances.append(text)

    return utterances


def parse_label_string(label_str: str) -> list:
    """Parse label string into list of integers."""
    # e.g., "[0 0 0 4 4]" or "[0, 0, 0, 4, 4]"

    try:
        # Try ast.literal_eval first
        labels = ast.literal_eval(label_str)
        if isinstance(labels, (list, tuple)):
            return [int(x) for x in labels]
    except:
        pass

    # Parse space or comma separated integers in brackets
    clean = label_str.strip('[]')
    parts = re.findall(r'\d+', clean)
    return [int(p) for p in parts]


def process_split(split_name: str, input_file: str, output_file: str):
    """Process a single split."""
    print(f"\nProcessing {split_name}...")

    df = pd.read_csv(input_file)
    print(f"  {len(df)} dialogues")

    records = []
    da_counts = {i: 0 for i in range(4)}
    emotion_counts = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        # Parse fields
        utterances = parse_dialog_string(str(row['dialog']))
        acts = parse_label_string(str(row['act']))
        emotions = parse_label_string(str(row['emotion']))

        # Verify lengths match
        n_utts = len(utterances)
        if len(acts) != n_utts:
            acts = acts[:n_utts] if len(acts) > n_utts else acts + [1] * (n_utts - len(acts))
        if len(emotions) != n_utts:
            emotions = emotions[:n_utts] if len(emotions) > n_utts else emotions + [0] * (n_utts - len(emotions))

        # Create utterance-level records
        for i, (utt, act, emo) in enumerate(zip(utterances, acts, emotions)):
            # Map emotion
            emo_str = DD_EMOTION_MAP.get(emo, 'neutral')
            emo_meld = EMOTION_TO_MELD.get(emo_str, 'neutral')

            # Map DA (1-4 -> 0-3)
            da = DD_DA_MAP.get(act, 0)

            # Alternate speakers (A, B, A, B, ...)
            speaker = 'A' if i % 2 == 0 else 'B'

            records.append({
                'file_id': f"dialog_{idx}",
                'utterance_id': i,
                'utterance': utt,
                'label': emo_meld,
                'speaker': speaker,
                'dialogue_act': da,
                'dialogue_act_name': DA_NAMES[da]
            })

            da_counts[da] += 1
            emotion_counts[emo_meld] = emotion_counts.get(emo_meld, 0) + 1

    # Create DataFrame and save
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_file, index=False)
    print(f"  Saved {len(out_df)} utterances to {output_file}")

    # Print statistics
    print(f"\n  Emotion distribution:")
    for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        print(f"    {emo}: {count} ({pct:.1f}%)")

    print(f"\n  Dialogue Act distribution:")
    for da_id, count in da_counts.items():
        pct = count / len(records) * 100
        print(f"    {DA_NAMES[da_id]}: {count} ({pct:.1f}%)")

    return out_df


def main():
    print("=" * 60)
    print("Preprocessing DailyDialog to Utterance-Level Format")
    print("=" * 60)

    # Process each split
    splits = [
        ('train', RAW_DIR / 'train.csv', OUTPUT_DIR / 'train_dailydialog.csv'),
        ('val', RAW_DIR / 'validation.csv', OUTPUT_DIR / 'val_dailydialog.csv'),
        ('test', RAW_DIR / 'test.csv', OUTPUT_DIR / 'test_dailydialog.csv'),
    ]

    all_stats = {}
    for split_name, input_file, output_file in splits:
        if not input_file.exists():
            print(f"  {input_file} not found, skipping")
            continue
        df = process_split(split_name, str(input_file), str(output_file))
        all_stats[split_name] = len(df)

    print("\n" + "=" * 60)
    print("Summary:")
    for split, count in all_stats.items():
        print(f"  {split}: {count} utterances")
    print("=" * 60)


if __name__ == "__main__":
    main()
