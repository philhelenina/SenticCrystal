import os
import pandas as pd
import glob
import re
import argparse

DATA_DIR = "/Volumes/ssd/corpora/IEMOCAP_full_release/"

ALL_EMOTIONS = ['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'sur', 'dis', 'fea', 'oth', 'xxx']
SIX_WAY_LABELS = ['ang', 'hap', 'sad', 'neu', 'exc', 'fru']
FOUR_WAY_LABELS = ['ang', 'hap', 'sad', 'neu']

LABEL_MAP = {
    'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral',
    'exc': 'Excited', 'fru': 'Frustrated', 'sur': 'Surprised', 'dis': 'Disgusted',
    'fea': 'Fearful', 'oth': 'Other', 'xxx': 'Undefined', '-1': 'Undefined'
}

TRAIN_SESSIONS = ['Session1', 'Session2', 'Session3']
VAL_SESSION = ['Session4']
TEST_SESSION = ['Session5']

def parse_emo_file(emo_file):
    emo_dict = {}
    with open(emo_file, 'r') as f:
        for line in f:
            if line.startswith('['):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    file_id = parts[1]
                    emotion = parts[2].lower()
                    emo_dict[file_id] = '-1' if emotion == 'xxx' else emotion
    return emo_dict

def process_line(line):
    match = re.match(r'(.*?)\s*\[(.*?)\]:\s*(.*)', line)
    if match:
        id_part = match.group(1).strip()
        time_range = match.group(2)
        utterance = match.group(3).strip()
        start, end = time_range.split('-')
        return {
            'id': id_part,
            'start': start.strip(),
            'end': end.strip(),
            'utterance': utterance
        }
    return None

def process_session(session_path, session_name):
    dialog_path = os.path.join(session_path, 'dialog')
    emo_files = glob.glob(os.path.join(dialog_path, 'EmoEvaluation', '*.txt'))
    trans_files = glob.glob(os.path.join(dialog_path, 'transcriptions', '*.txt'))

    print(f"Found {len(emo_files)} emotion files and {len(trans_files)} transcription files in {session_name}")

    all_data = []
    file_count = 0
    skipped_files = []

    for trans_file in trans_files:
        file_name = os.path.basename(trans_file)
        emo_file = os.path.join(dialog_path, 'EmoEvaluation', file_name)

        if not os.path.exists(emo_file):
            print(f"Warning: No matching EmoEvaluation file found for {trans_file}")
            print(f"Expected EmoEvaluation file: {emo_file}")
            skipped_files.append(trans_file)
            continue

        print(f"Processing file: {file_name}")

        emo_dict = parse_emo_file(emo_file)

        with open(trans_file, 'r') as f:
            lines = f.readlines()

        file_data = []
        utterance_num = 0

        for line in lines:
            processed = process_line(line)
            if processed:
                emotion = emo_dict.get(processed['id'], '-1')
                utterance_num += 1
                file_data.append({
                    'session': session_name,
                    'utterance_num': utterance_num,
                    'id': processed['id'],
                    'start': processed['start'],
                    'end': processed['end'],
                    'utterance': processed['utterance'],
                    'label': emotion
                })

        if file_data:
            all_data.extend(file_data)
            file_count += 1
            print(f"File: {file_name}, Utterances processed: {len(file_data)}")
        else:
            print(f"Warning: No data processed for file {trans_file}")
            skipped_files.append(trans_file)

    print(f"Processed {file_count} files in {session_name}")
    if skipped_files:
        print(f"Skipped files in {session_name}:")
        for file in skipped_files:
            print(f"  - {file}")

    return all_data

def process_and_save_sessions(sessions, data_dir, prefix):
    all_data = []

    for session in sessions:
        session_path = os.path.join(data_dir, session)
        session_data = process_session(session_path, session)
        all_data.extend(session_data)

    if not all_data:
        print(f"Warning: No data processed for {prefix}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Extracting the file_id in the format like Ses01M_impro01 or Ses01M_script01
    df['file_id'] = df['id'].str.extract(r'(Ses\d+[FM]_(?:impro|script)\d+)')

    # Grouping by file_id and assigning the cumulative utterance number within each file
    df['utterance_num'] = df.groupby('file_id').cumcount()

    # Sorting the DataFrame by file_id and utterance_num
    df = df.sort_values(by=['file_id', 'utterance_num']).reset_index(drop=True)

    # Saving the results
    df.to_csv(f'{prefix}_full_data.csv', index=False)
    df[['id', 'utterance_num', 'label']].to_csv(f'{prefix}_metadata.csv', index=False)
    df[['id', 'utterance']].to_csv(f'{prefix}_utterances.csv', index=False)

    return df


def create_classification_data(df, classification_type, use_minus_one):
    if classification_type == '4':
        df['label'] = df['label'].apply(lambda x: 'hap' if x == 'exc' else x)
        valid_labels = FOUR_WAY_LABELS
    elif classification_type == '6':
        valid_labels = SIX_WAY_LABELS
    else:
        raise ValueError("Invalid classification type. Use '4' or '6'.")

    if use_minus_one:
        df['label'] = df['label'].apply(lambda x: x if x in valid_labels else '-1')
    else:
        df = df[df['label'].isin(valid_labels)]

    label_to_num = {label: i for i, label in enumerate(valid_labels)}
    df['label_num'] = df['label'].map(label_to_num)

    return df

def print_statistics(df, title, classification_type):
    print(f"\n{title}:")
    print(f"Total utterances: {len(df)}")
    print("\nLabel distribution:")

    valid_labels = FOUR_WAY_LABELS if classification_type == '4' else SIX_WAY_LABELS
    label_counts = df['label'].value_counts().sort_index()

    for i, label in enumerate(valid_labels):
        count = label_counts.get(label, 0)
        print(f"{LABEL_MAP[label]} ({i})\t--> {count}")

    if '-1' in label_counts:
        print(f"Undefined (-1)\t--> {label_counts['-1']}")

    print(f"\nTotal unique speakers: {df['id'].str[:5].nunique()}")
    print(f"Total unique sessions: {df['session'].nunique()}")
    print(f"Files per session: {df.groupby('session')['id'].nunique().to_dict()}")

def main():
    parser = argparse.ArgumentParser(description="Process IEMOCAP dataset")
    parser.add_argument('--use_minus_one', action='store_true', help="Use -1 for non-selected labels")
    args = parser.parse_args()

    train_data = process_and_save_sessions(TRAIN_SESSIONS, DATA_DIR, 'train')
    val_data = process_and_save_sessions(VAL_SESSION, DATA_DIR, 'val')
    test_data = process_and_save_sessions(TEST_SESSION, DATA_DIR, 'test')

    if not train_data.empty and not val_data.empty and not test_data.empty:
        all_data = pd.concat([train_data, val_data, test_data])
        print_statistics(all_data, "Overall Statistics", '6')

        for classification_type in ['4', '6']:
            print(f"\nProcessing {classification_type}-way classification")
            train_classified = create_classification_data(train_data, classification_type, args.use_minus_one)
            val_classified = create_classification_data(val_data, classification_type, args.use_minus_one)
            test_classified = create_classification_data(test_data, classification_type, args.use_minus_one)

            all_classified = pd.concat([train_classified, val_classified, test_classified])
            print_statistics(all_classified, f"Statistics for {classification_type}-way classification", classification_type)

            print(f"\nTraining set size: {len(train_classified)}")
            print(f"Validation set size: {len(val_classified)}")
            print(f"Test set size: {len(test_classified)}")

            train_classified.to_csv(f'train_{classification_type}way{"_with_minus_one" if args.use_minus_one else ""}.csv', index=False)
            val_classified.to_csv(f'val_{classification_type}way{"_with_minus_one" if args.use_minus_one else ""}.csv', index=False)
            test_classified.to_csv(f'test_{classification_type}way{"_with_minus_one" if args.use_minus_one else ""}.csv', index=False)
    else:
        print("Error: Failed to process data")

if __name__ == "__main__":
    main()

# python preprocessing.py --use_minus_one
