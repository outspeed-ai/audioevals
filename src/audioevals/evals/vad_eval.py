import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from audioevals.utils.audio import AudioData
from audioevals.utils.vad.silero import SileroVADModel, calculate_rms


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
            "average_max_silence_duration": float,  # in seconds
            "results": [
                {
                    "audio_file": str,
                    "max_silence_duration": float,  # in seconds
                    "silence_to_speech_ratio": float,  # total_silence_duration / total_speech_duration
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
        "average_max_silence_duration": 0.0,
        "results": [],
    }

    print("Starting VAD evaluation...")
    print(f"Audio directory: {audio_path}")
    print(f"Total files to evaluate: {len(ground_truth_transcripts)}")
    print("-" * 50)

    # Initialize Silero VAD model for 16kHz
    vad_model = SileroVADModel(sample_rate=16000)
    
    max_silence_durations = []
    silence_to_speech_ratios = []

    # Process each audio file
    for audio_filename, _ in ground_truth_transcripts.items():
        audio_file = audio_path / audio_filename

        result = {
            "audio_file": str(audio_file),
            "max_silence_duration": 0.0,
            "silence_to_speech_ratio": 0.0,
        }

        try:
            print(f"Processing {audio_filename}...")

            audio_data = AudioData.from_wav_file(str(audio_file))
            
            # Resample to 16kHz if needed
            if audio_data.sample_rate != 16000:
                audio_data = audio_data.resample(16000)
            
            audio_array = audio_data.get_1d_array(np.float32)
            
            vad_model.reset_states()
            
            chunk_size = 512
            speech_detection_results = []
            
            # Pad audio if needed
            if len(audio_array) % chunk_size != 0:
                padding_size = chunk_size - (len(audio_array) % chunk_size)
                audio_array = np.pad(audio_array, (0, padding_size), mode='constant')
            
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    vad_score = vad_model(chunk)
                    rms_value = calculate_rms(chunk)
                    
                    is_speech = vad_score >= 0.5 and rms_value >= 0.001
                    speech_detection_results.append(is_speech)
            
            # Calculate silence and speech durations
            max_silence_duration, silence_to_speech_ratio = calculate_silence_metrics(
                speech_detection_results, chunk_duration_ms=32
            )
            
            result["max_silence_duration"] = max_silence_duration
            result["silence_to_speech_ratio"] = silence_to_speech_ratio
            
            max_silence_durations.append(max_silence_duration)
            silence_to_speech_ratios.append(silence_to_speech_ratio)
            results["successful_evaluations"] += 1
            
            print(f"  ✅ Max silence: {max_silence_duration:.2f}s | Silence/Speech ratio: {silence_to_speech_ratio:.2f}")

        except Exception as e:
            error_msg = str(e)
            results["failed_evaluations"] += 1
            print(f"  ❌ Failed: {error_msg}")

        results["results"].append(result)

    # Calculate average max silence duration
    if max_silence_durations:
        results["average_max_silence_duration"] = sum(max_silence_durations) / len(max_silence_durations)

    # Print summary
    print("-" * 50)
    print("VAD Evaluation Complete!")
    print(f"✅ Evaluated: {results['successful_evaluations']}/{results['total_files']}")
    if max_silence_durations:
        print(f"📊 Average max silence duration: {results['average_max_silence_duration']:.2f}s")
        print(f"📊 Shortest max silence: {min(max_silence_durations):.2f}s")
        print(f"📊 Longest max silence: {max(max_silence_durations):.2f}s")
    if silence_to_speech_ratios:
        print(f"📊 Average silence/speech ratio: {sum(silence_to_speech_ratios) / len(silence_to_speech_ratios):.2f}")
    print("-" * 50)

    return results


def calculate_silence_metrics(speech_detection_results: List[bool], chunk_duration_ms: int = 32) -> tuple[float, float]:
    if not speech_detection_results:
        return 0.0, 0.0
    
    # Find maximum continuous silence duration
    max_silence_chunks = 0
    current_silence_chunks = 0
    
    for is_speech in speech_detection_results:
        if not is_speech:  # Silence
            current_silence_chunks += 1
            max_silence_chunks = max(max_silence_chunks, current_silence_chunks)
        else:  # Speech
            current_silence_chunks = 0
    
    total_silence_chunks = sum(1 for is_speech in speech_detection_results if not is_speech)
    total_speech_chunks = sum(1 for is_speech in speech_detection_results if is_speech)
    
    max_silence_duration = (max_silence_chunks * chunk_duration_ms) / 1000.0
    
    # silence to speech ratio
    if total_speech_chunks > 0:
        silence_to_speech_ratio = total_silence_chunks / total_speech_chunks
    else:
        silence_to_speech_ratio = float('inf') if total_silence_chunks > 0 else 0.0
    
    return max_silence_duration, silence_to_speech_ratio