import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from audioevals.utils.audio import AudioData
from audioevals.utils.nisqa.nisqa import Nisqa


def run(
    audio_dir: str,
    transcripts_file: Optional[str] = None,
) -> Dict:
    """
    Returns:
        {
            "total_files": int,
            "successful_evaluations": int,
            "failed_evaluations": int,
            "average_mos": float,        # average MOS score (1-5 scale)
            "results": [
                {
                    "audio_file": str,
                    "mos_score": float,      # NISQA MOS prediction (1-5)
                }
            ]
        }
    """

    if transcripts_file is None:
        transcripts_path = Path(__file__).parent.parent / "transcripts.json"
    else:
        transcripts_path = Path(transcripts_file)

    with open(transcripts_path, "r") as f:
        ground_truth_transcripts: Dict[str, str] = json.load(f)

    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_path}")

    results = {
        "total_files": len(ground_truth_transcripts),
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "average_mos": 0.0,
        "results": [],
    }

    print("Starting NISQA evaluation...")
    print(f"Audio directory: {audio_path}")
    print(f"Total files to evaluate: {len(ground_truth_transcripts)}")
    print("-" * 50)

    mos_scores = []

    # Process each audio file
    for audio_filename, _ in ground_truth_transcripts.items():
        audio_file = audio_path / audio_filename

        result = {
            "audio_file": str(audio_file),
            "mos_score": 0.0,
        }

        try:
            print(f"Processing {audio_filename}...")

            audio_data = AudioData.from_wav_file(str(audio_file))

            # Delegate to run_audio_data for processing
            nisqa_result = run_audio_data(audio_data)

            result["mos_score"] = nisqa_result["mos_score"]

            mos_scores.append(nisqa_result["mos_score"])
            results["successful_evaluations"] += 1

            print(f"  ✅ MOS score: {nisqa_result['mos_score']:.2f}")

        except Exception as e:
            error_msg = str(e)
            results["failed_evaluations"] += 1
            print(f"  ❌ Failed: {error_msg}")

        results["results"].append(result)

    # Calculate average MOS
    if mos_scores:
        results["average_mos"] = sum(mos_scores) / len(mos_scores)

    # Print summary
    print("-" * 50)
    print("NISQA Evaluation Complete!")
    print(f"✅ Evaluated: {results['successful_evaluations']}/{results['total_files']}")
    if mos_scores:
        print(f"📊 Average MOS: {results['average_mos']:.2f}")
        print(f"📊 MOS range: {min(mos_scores):.2f} - {max(mos_scores):.2f}")
        print("📊 Quality distribution:")
        excellent = sum(1 for s in mos_scores if s >= 4.0)
        good = sum(1 for s in mos_scores if 3.0 <= s < 4.0)
        fair = sum(1 for s in mos_scores if 2.0 <= s < 3.0)
        poor = sum(1 for s in mos_scores if s < 2.0)
        print(f"    Excellent (≥4.0): {excellent}")
        print(f"    Good (3.0-3.99): {good}")
        print(f"    Fair (2.0-2.99): {fair}")
        print(f"    Poor (<2.0): {poor}")
    print("-" * 50)

    return results


def run_single_file(audio_file_path: str) -> Dict:
    """
    Returns:
        {
            "mos_score": float,
        }
    """
    audio_data = AudioData.from_wav_file(audio_file_path)
    return run_audio_data(audio_data)


def run_audio_data(audio_data: AudioData) -> Dict:
    """
    Returns:
        {
            "mos_score": float,
        }
    """
    result = {
        "mos_score": 0.0,
    }

    # Get audio as float32 for NISQA processing
    audio_array = audio_data.get_1d_array(np.float32)

    # NISQA expects 48kHz audio - resample if needed
    if audio_data.sample_rate != 48000:
        # Resample to 48kHz
        audio_data_48k = audio_data.resample(48000)
        audio_array = audio_data_48k.get_1d_array(np.float32)
        sample_rate = 48000
    else:
        sample_rate = audio_data.sample_rate

    # Run NISQA prediction
    # Use the default TTS model weights
    pretrained_model_path = (
        Path(__file__).parent.parent / "utils" / "nisqa" / "weights" / "nisqa_tts.tar"
    )

    try:
        mos_score = Nisqa.run_audio_data(
            audio_array,
            sample_rate=sample_rate,
            pretrained_model=str(pretrained_model_path),
        )
        result["mos_score"] = float(mos_score)
    except Exception as e:
        # If pretrained model not found, try without specifying path (use default)
        if "not found" in str(e):
            print(
                f"  ⚠️  Default model not found at {pretrained_model_path}, trying fallback..."
            )
            mos_score = Nisqa.run_audio_data(audio_array, sample_rate=sample_rate)
            result["mos_score"] = float(mos_score)
        else:
            raise e

    return result
