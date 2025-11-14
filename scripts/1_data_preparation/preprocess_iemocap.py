"""
SOTA-compliant preprocessing for IEMOCAP dataset.

Based on latest SOTA papers:
- IEMOCAP 4way: angry, happy (exc+hap merged), sad, neutral
- IEMOCAP 6way: angry, happy, sad, neutral, excited, frustrated

All utterances are kept for context modeling (unlabeled utterances have label_num=-1).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
import glob
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def process_iemocap_raw(
    iemocap_raw_path: str,
    output_path: str,
    classification_type: str = '4way'
):
    """
    Process raw IEMOCAP data following SOTA standards.

    IMPORTANT: For context modeling, ALL utterances are kept (including unlabeled).

    SOTA splits:
    - Train: Session2, Session3, Session4
    - Val: Session1
    - Test: Session5

    4way: angry, happy, sad, neutral (excited->happy, frustrated NOT included)
    6way: angry, happy, sad, neutral, excited, frustrated

    Unlabeled utterances (xxx, fea, sur, unannotated, etc.) are kept with label_num=-1 for context.
    """
    
    iemocap_path = Path(iemocap_raw_path)
    output_dir = Path(output_path) / 'iemocap' / classification_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SOTA session splits (following paper specification)
    # Paper: "sessions 2-4 for training, session 1 for validation, session 5 for testing"
    session_splits = {
        'train': ['Session2', 'Session3', 'Session4'],
        'val': ['Session1'],
        'test': ['Session5']
    }
    
    # SOTA emotion mappings
    if classification_type == '4way':
        # Paper: "angry, happy, sad, neutral and excited categories (with excited and happy categories merged)"
        # IMPORTANT: frustrated is NOT included in 4way (only in 6way)
        emotion_map = {
            'ang': 'angry',
            'hap': 'happy',
            'sad': 'sad',
            'neu': 'neutral',
            'exc': 'happy',    # SOTA: excited -> happy
            # 'fru' is excluded (not mapped) - will become 'undefined'
            # All others map to 'undefined' and get filtered out
        }
        target_labels = ['angry', 'happy', 'sad', 'neutral']
    else:  # 6way
        emotion_map = {
            'ang': 'angry',
            'hap': 'happy',
            'sad': 'sad', 
            'neu': 'neutral',
            'exc': 'excited',
            'fru': 'frustrated',
            # All others map to -1 (undefined)
        }
        target_labels = ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    
    label_to_num = {label: i for i, label in enumerate(target_labels)}
    
    # Process each split
    for split, sessions in session_splits.items():
        all_data = []

        for session in sessions:
            logger.info(f"Processing {session} for {split}")
            session_path = iemocap_path / session / 'dialog'

            # Get all transcription files
            trans_files = glob.glob(str(session_path / 'transcriptions' / '*.txt'))

            for trans_file in trans_files:
                file_name = Path(trans_file).name
                emo_file = session_path / 'EmoEvaluation' / file_name

                # Parse emotion file (if exists)
                emo_dict = {}
                if emo_file.exists():
                    with open(emo_file, 'r') as f:
                        for line in f:
                            if line.startswith('['):
                                parts = line.strip().split('\t')
                                if len(parts) >= 3:
                                    utterance_id = parts[1]
                                    emotion = parts[2].lower()
                                    emo_dict[utterance_id] = emotion
                else:
                    logger.warning(f"  No emotion file for {file_name}")

                # Parse transcription file (ALL utterances for context modeling)
                with open(trans_file, 'r') as f:
                    for line in f:
                        # Format: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
                        match = re.match(r'(\S+)\s+\[([0-9.]+)-([0-9.]+)\]:\s*(.*)', line.strip())
                        if match:
                            utterance_id = match.group(1)
                            start_time = match.group(2)
                            end_time = match.group(3)
                            utterance = match.group(4)

                            # Get emotion (if annotated)
                            emotion = emo_dict.get(utterance_id, None)

                            if emotion is None:
                                # Not annotated (transcription only)
                                mapped_emotion = 'unannotated'
                                original_emotion = 'unannotated'
                                label_num = -1
                            elif emotion in emotion_map:
                                # Target emotion (labeled)
                                mapped_emotion = emotion_map[emotion]
                                original_emotion = emotion
                                label_num = label_to_num[mapped_emotion]
                            else:
                                # Other emotion (xxx, fea, sur, etc.) - unlabeled but keep for context
                                mapped_emotion = emotion
                                original_emotion = emotion
                                label_num = -1

                            # Extract file_id
                            file_id = utterance_id[:15]  # e.g., Ses01F_impro01

                            all_data.append({
                                'id': utterance_id,
                                'utterance': utterance,
                                'label': mapped_emotion,
                                'label_num': label_num,
                                'file_id': file_id,
                                'start': start_time,
                                'end': end_time,
                                'session': session,
                                'original_emotion': original_emotion,
                                'utterance_num': 0  # Will be recalculated
                            })

        if all_data:
            df = pd.DataFrame(all_data)

            # Recalculate utterance_num within each file (for conversation order)
            df['utterance_num'] = df.groupby('file_id').cumcount()

            # Sort by file_id and utterance_num
            df = df.sort_values(['file_id', 'utterance_num']).reset_index(drop=True)

            # Save ALL utterances (including unlabeled for context modeling)
            main_path = output_dir / f"{split}_{classification_type}_unified.csv"
            df.to_csv(main_path, index=False)

            # Save metadata (all utterances)
            meta_df = df[['id', 'utterance_num', 'label', 'label_num']].copy()
            meta_path = output_dir / f"{split}_{classification_type}_metadata.csv"
            meta_df.to_csv(meta_path, index=False)

            # Save utterances only
            utt_df = df[['id', 'utterance']].copy()
            utt_path = output_dir / f"{split}_{classification_type}_utterances.csv"
            utt_df.to_csv(utt_path, index=False)

            # Print statistics
            labeled_df = df[df['label_num'] != -1]
            unlabeled_df = df[df['label_num'] == -1]
            unannotated_df = df[df['label'] == 'unannotated']

            logger.info(f"  {split.upper()}: {len(df)} total utterances")
            logger.info(f"    Labeled ({classification_type}): {len(labeled_df)}")
            logger.info(f"    Unlabeled (context): {len(unlabeled_df)}")
            logger.info(f"      - Unannotated: {len(unannotated_df)}")
            logger.info(f"      - Other emotions: {len(unlabeled_df) - len(unannotated_df)}")

            logger.info(f"  Label distribution (labeled only):")
            label_counts = labeled_df['label'].value_counts()
            for label in target_labels:
                count = label_counts.get(label, 0)
                percentage = (count / len(labeled_df)) * 100 if len(labeled_df) > 0 else 0
                logger.info(f"    {label}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="IEMOCAP preprocessing for SOTA-compliant emotion recognition")
    parser.add_argument('--classification', choices=['4way', '6way', 'both'], default='both',
                       help='Classification type: 4way, 6way, or both')
    parser.add_argument('--iemocap_raw', type=str,
                       default='/Volumes/ssd/corpora/IEMOCAP_full_release',
                       help='Path to raw IEMOCAP dataset')
    parser.add_argument('--output', type=str,
                       default='/Volumes/ssd/01-ckj-postdoc/Lingua-Emoca/SenticCrystal/data',
                       help='Output directory for processed data')

    args = parser.parse_args()

    classification_types = ['4way', '6way'] if args.classification == 'both' else [args.classification]

    # Process IEMOCAP
    for classification_type in classification_types:
        logger.info(f"Processing IEMOCAP {classification_type}")
        process_iemocap_raw(args.iemocap_raw, args.output, classification_type)

    logger.info("IEMOCAP preprocessing completed!")

if __name__ == "__main__":
    main()