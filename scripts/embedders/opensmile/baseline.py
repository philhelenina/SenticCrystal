#!/usr/bin/env python3
"""Aggregate eGeMAPSv02 openSMILE embeddings to speaker-level summaries.

The input embeddings are expected to be utterance-level eGeMAPSv02 vectors with
shape (n, 1, 88) or any equivalent shape that can be flattened to (n, 88).

The script groups rows by speaker ID, where the speaker ID is the input CSV
`id` value with the last three characters removed, and computes one output row
per speaker.

The 88-dimensional vector is treated as the standard eGeMAPSv02 block layout:
pitch/F0, loudness, speech-rate/temporal, and voice-quality/spectral blocks.
Because the raw embeddings do not carry feature names, the block boundaries are
encoded explicitly below and can be adjusted in one place if a different openSMILE
build or feature ordering is used.

Example usage:
    python scripts/embedders/opensmile/baseline.py \
        data/embeddings/4way/opensmile/eGeMAPSv02/test_unified_filtered.npy \
        data/iemocap_4way_data/test_unified_filtered.csv \
        data/iemocap_4way_data/test_speaker_baseline.csv

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DIM = 88

# Block layout for the standard eGeMAPSv02 functionals output.
# The slices are intentionally coarse-grained: they aggregate the feature blocks
# that correspond to the requested speaker descriptors.
F0_BLOCK = slice(0, 21)
LOUDNESS_BLOCK = slice(21, 42)
SPEECH_RATE_BLOCK = slice(42, 48)
VOICE_QUALITY_BLOCK = slice(48, 88)

# Within the F0 block, the first five values are the most direct pitch summary
# descriptors (mean, dispersion, percentiles / range-like statistics).
F0_MEAN_INDICES = [0]
F0_RANGE_INDICES = [4, 5]

# Loudness is summarized from the same type of low-order loudness descriptors.
LOUDNESS_INDICES = [0]

# Speech-rate is approximated from the temporal / segment-rate descriptors.
SPEECH_RATE_INDICES = [0, 1, 2, 3, 4, 5]

# Voice quality is summarized from jitter, shimmer, harmonicity, and formant-related
# descriptors in the voiced block.
VOICE_QUALITY_INDICES = list(range(0, 40))


def load_embeddings(path: Path) -> np.ndarray:
	embeddings = np.load(path)
	embeddings = np.asarray(embeddings, dtype=np.float32)

	if embeddings.ndim == 1:
		embeddings = embeddings.reshape(1, -1)
	elif embeddings.ndim == 3:
		embeddings = embeddings.reshape(embeddings.shape[0], -1)

	if embeddings.ndim != 2:
		raise ValueError(f"Expected embeddings with 2 or 3 dimensions, got shape {embeddings.shape}")

	if embeddings.shape[1] != EXPECTED_DIM:
		if embeddings.shape[0] == EXPECTED_DIM and embeddings.shape[1] != EXPECTED_DIM:
			embeddings = embeddings.T
		else:
			raise ValueError(
				f"Expected embeddings with dimension {EXPECTED_DIM}, got shape {embeddings.shape}"
			)

	return embeddings


def speaker_id_from_utterance_id(utterance_id: str) -> str:
	return utterance_id[:-3]


def mean_over_indices(row: np.ndarray, indices: list[int]) -> float:
	return float(np.nanmean(row[indices]))


def summarize_row(row: np.ndarray) -> dict[str, float]:
	return {
		"f0_mean": mean_over_indices(row[F0_BLOCK], F0_MEAN_INDICES),
		"f0_range": mean_over_indices(row[F0_BLOCK], F0_RANGE_INDICES),
		"loudness": mean_over_indices(row[LOUDNESS_BLOCK], LOUDNESS_INDICES),
		"speech_rate": mean_over_indices(row[SPEECH_RATE_BLOCK], SPEECH_RATE_INDICES),
		"voice_quality": mean_over_indices(row[VOICE_QUALITY_BLOCK], VOICE_QUALITY_INDICES),
	}


def build_output(embeddings_path: Path, csv_path: Path) -> pd.DataFrame:
	embeddings = load_embeddings(embeddings_path)
	metadata = pd.read_csv(csv_path)

	required_columns = {"id", "utterance"}
	missing = required_columns.difference(metadata.columns)
	if missing:
		raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

	if len(metadata) != len(embeddings):
		raise ValueError(
			f"Row count mismatch: CSV has {len(metadata)} rows but embeddings have {len(embeddings)} rows"
		)

	metadata = metadata.copy()
	metadata["speaker_id"] = metadata["id"].astype(str).map(speaker_id_from_utterance_id)

	rows = []
	for speaker_id, speaker_frame in metadata.groupby("speaker_id", sort=False):
		speaker_indices = speaker_frame.index.to_numpy()
		speaker_embeddings = embeddings[speaker_indices]
		speaker_vectors = np.array([list(summarize_row(row).values()) for row in speaker_embeddings], dtype=np.float32)
		speaker_summary = np.nanmean(speaker_vectors, axis=0)
		rows.append(
			{
				"speaker_id": speaker_id,
				"f0_mean": float(speaker_summary[0]),
				"f0_range": float(speaker_summary[1]),
				"loudness": float(speaker_summary[2]),
				"speech_rate": float(speaker_summary[3]),
				"voice_quality": float(speaker_summary[4]),
			}
		)

	return pd.DataFrame(rows, columns=["speaker_id", "f0_mean", "f0_range", "loudness", "speech_rate", "voice_quality"])


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Aggregate utterance-level eGeMAPSv02 embeddings to speaker-level summaries"
	)
	parser.add_argument("embeddings_npy", type=Path, help="Path to the npy file of embeddings")
	parser.add_argument("input_csv", type=Path, help="Path to the CSV file with the original text")
	parser.add_argument("output_csv", type=Path, help="Path to the output CSV file")
	args = parser.parse_args()

	output = build_output(args.embeddings_npy, args.input_csv)
	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	output.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
	main()
