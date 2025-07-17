import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from audioevals.utils.common import get_similarity_score, normalize_text, request_deepgram_stt
from audioevals.utils.audio import AudioData


async def run(
    audio_dir: str,
    transcripts_file: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Returns:
        {
            "total_files": int,
            "successful_evaluations": int,
            "failed_evaluations": int,
            "results": [
                {
                    "file_index": int,
                    "audio_file": str,
                    "stt_transcript": str,
                    "wer_score": float,
                }
            ]
        }
    """

    # Load ground truth transcripts
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
        "average_wer": 0.0,
        "results": [],
    }

    print("Starting WER evaluation...")
    print(f"Audio directory: {audio_path}")
    print(f"Total files to evaluate: {len(ground_truth_transcripts)}")
    print("-" * 50)

    wer_scores = []

    # Process each audio file
    for audio_filename, ground_truth in ground_truth_transcripts.items():
        audio_file = audio_path / audio_filename

        result = {
            "audio_file": str(audio_file),
            "stt_transcript": "",
            "wer_score": 0.0,
        }

        try:
            print(f"Processing {audio_filename}: {ground_truth[:50]}...")

            # Check if audio file exists
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            # Load audio data
            audio_data = AudioData.from_wav_file(str(audio_file))

            wer_result = await run_audio_data(audio_data, ground_truth)

            result["stt_transcript"] = wer_result["stt_transcript"]
            result["wer_score"] = wer_result["wer_score"]

            wer_scores.append(wer_result["wer_score"])
            results["successful_evaluations"] += 1

            print(f'  ✅ WER: {wer_result["wer_score"]:.2f}% | STT: "{wer_result["stt_transcript"]}"')

        except Exception as e:
            error_msg = str(e)
            results["failed_evaluations"] += 1

            print(f"  ❌ Failed: {error_msg}")

        results["results"].append(result)

    # Calculate average WER
    if wer_scores:
        results["average_wer"] = sum(wer_scores) / len(wer_scores)

    # Print summary
    print("-" * 50)
    print("WER Evaluation Complete!")
    print(f"✅ Evaluated: {results['successful_evaluations']}/{results['total_files']}")
    if wer_scores:
        print(f"📊 Average WER: {results['average_wer']:.2f}%")
        print(f"📊 Best WER: {min(wer_scores):.2f}%")
        print(f"📊 Worst WER: {max(wer_scores):.2f}%")
    print("-" * 50)

    return results


async def run_single_file(audio_file_path: str, ground_truth_transcript: str) -> Dict:
    """
    Returns:
        {
            "stt_transcript": str,
            "wer_score": float,
        }
    """
    audio_data = AudioData.from_wav_file(audio_file_path)
    return await run_audio_data(audio_data, ground_truth_transcript)


async def run_audio_data(audio_data: AudioData, ground_truth_transcript: str) -> Dict:
    """
    Returns:
        {
            "stt_transcript": str,
            "wer_score": float,
        }
    """
    result = {
        "stt_transcript": "",
        "wer_score": 0.0,
    }
    
    # Run STT
    stt_transcript = await request_deepgram_stt(audio_data)
    
    if stt_transcript is None:
        raise Exception("STT returned None - transcription failed")
    
    result["stt_transcript"] = normalize_text(stt_transcript)
    
    # Calculate WER
    wer_score = get_similarity_score(ground_truth_transcript, stt_transcript)
    result["wer_score"] = wer_score
    
    return result