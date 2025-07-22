import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import parselmouth

sys.path.append(str(Path(__file__).parent.parent.parent))

from audioevals.utils.audio import AudioData


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
            "average_mean_pitch": float,  # in Hz
            "average_pitch_std": float,   # in Hz
            "results": [
                {
                    "audio_file": str,
                    "mean_pitch": float,      # in Hz
                    "pitch_std": float,       # in Hz
                    "pitch_min": float,       # in Hz
                    "pitch_max": float,       # in Hz
                    "pitch_range": float,     # in Hz
                    "pitch_values": list,     # array of pitch frequency values (Hz)
                    "frame_size_ms": float,   # frame size in milliseconds
                    "pitch_stability": float, # coefficient of variation (CV)
                    "semitone_jumps": int,    # number of large semitone jumps
                    "octave_errors": int,     # number of octave-level errors
                    "max_semitone_jump": float, # largest semitone jump detected
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
        "average_mean_pitch": 0.0,
        "average_pitch_std": 0.0,
        "results": [],
    }

    print("Starting pitch evaluation...")
    print(f"Audio directory: {audio_path}")
    print(f"Total files to evaluate: {len(ground_truth_transcripts)}")
    print("-" * 50)

    mean_pitches = []
    pitch_stds = []

    for audio_filename, _ in ground_truth_transcripts.items():
        audio_file = audio_path / audio_filename

        result = {
            "audio_file": str(audio_file),
            "mean_pitch": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "pitch_range": 0.0,
            "pitch_values": [],
            "frame_size_ms": 0.0,
            "pitch_stability": 0.0,
            "semitone_jumps": 0,
            "octave_errors": 0,
            "max_semitone_jump": 0.0,
        }

        try:
            print(f"Processing {audio_filename}...")

            audio_data = AudioData.from_wav_file(str(audio_file))

            pitch_result = run_audio_data(audio_data)

            result.update(pitch_result)

            mean_pitches.append(pitch_result["mean_pitch"])
            pitch_stds.append(pitch_result["pitch_std"])
            results["successful_evaluations"] += 1

            print(
                f"  ✅ Mean pitch: {pitch_result['mean_pitch']:.1f}Hz | CV: {pitch_result['pitch_stability']:.2f} | Jumps: {pitch_result['semitone_jumps']} | Octave errors: {pitch_result['octave_errors']}"
            )

        except Exception as e:
            error_msg = str(e)
            results["failed_evaluations"] += 1
            print(f"  ❌ Failed: {error_msg}")

        results["results"].append(result)

    # Calculate averages
    if mean_pitches:
        results["average_mean_pitch"] = sum(mean_pitches) / len(mean_pitches)
        results["average_pitch_std"] = sum(pitch_stds) / len(pitch_stds)

    # Print summary
    print("-" * 50)
    print("Pitch Evaluation Complete!")
    print(f"✅ Evaluated: {results['successful_evaluations']}/{results['total_files']}")
    if mean_pitches:
        print(f"📊 Average mean pitch: {results['average_mean_pitch']:.1f}Hz")
        print(f"📊 Average pitch std: {results['average_pitch_std']:.1f}Hz")
        print(f"📊 Pitch range: {min(mean_pitches):.1f}Hz - {max(mean_pitches):.1f}Hz")
    print("-" * 50)

    return results


def run_single_file(audio_file_path: str, time_step: Optional[float] = None) -> Dict:
    """
    Returns:
        {
            "mean_pitch": float,
            "pitch_std": float,
            "pitch_min": float,
            "pitch_max": float,
            "pitch_range": float,
            "pitch_values": list,
            "frame_size_ms": float,
            "pitch_stability": float,
            "semitone_jumps": int,
            "octave_errors": int,
            "max_semitone_jump": float,
        }
    """
    audio_data = AudioData.from_wav_file(audio_file_path)
    return run_audio_data(audio_data, time_step)


def run_audio_data(audio_data: AudioData, time_step: Optional[float] = None) -> Dict:
    """
    Returns:
        {
            "mean_pitch": float,
            "pitch_std": float,
            "pitch_min": float,
            "pitch_max": float,
            "pitch_range": float,
            "pitch_values": list,
            "frame_size_ms": float,
            "pitch_stability": float,
            "semitone_jumps": int,
            "octave_errors": int,
            "max_semitone_jump": float,
        }
    """
    result = {
        "mean_pitch": 0.0,
        "pitch_std": 0.0,
        "pitch_min": 0.0,
        "pitch_max": 0.0,
        "pitch_range": 0.0,
        "pitch_values": [],
        "frame_size_ms": 0.0,
        "pitch_stability": 0.0,
        "semitone_jumps": 0,
        "octave_errors": 0,
        "max_semitone_jump": 0.0,
    }

    audio_array = audio_data.get_1d_array(np.float32)

    sound = parselmouth.Sound(audio_array, sampling_frequency=audio_data.sample_rate)
    if time_step is None:
        pitch = sound.to_pitch()
    else:
        pitch = sound.to_pitch(time_step=time_step)

    pitch_values = pitch.selected_array["frequency"]
    result["pitch_values"] = pitch_values.tolist()

    # Calculate frame size in ms
    if len(pitch_values) > 0:
        result["frame_size_ms"] = pitch.time_step * 1000.0

    voiced_pitch_values = pitch_values[pitch_values > 0]

    if len(voiced_pitch_values) > 0:
        result["mean_pitch"] = float(np.mean(voiced_pitch_values))
        result["pitch_std"] = float(np.std(voiced_pitch_values))
        result["pitch_min"] = float(np.min(voiced_pitch_values))
        result["pitch_max"] = float(np.max(voiced_pitch_values))
        result["pitch_range"] = result["pitch_max"] - result["pitch_min"]

        # Calculate pitch stability using coefficient of variation
        if result["mean_pitch"] > 0:
            cv = result["pitch_std"] / result["mean_pitch"]
            result["pitch_stability"] = float(cv)

        # Calculate semitone jumps to detect octave errors
        semitone_jumps, octave_errors, max_jump = _calculate_semitone_jumps(
            voiced_pitch_values
        )
        result["semitone_jumps"] = semitone_jumps
        result["octave_errors"] = octave_errors
        result["max_semitone_jump"] = max_jump

    return result


def _calculate_semitone_jumps(pitch_values: np.ndarray) -> tuple[int, int, float]:
    """
    Calculate semitone jumps between consecutive pitch values to detect artifacts.

    Args:
        pitch_values: Array of pitch values in Hz (only voiced frames)

    Returns:
        tuple: (semitone_jumps, octave_errors, max_semitone_jump)
            - semitone_jumps: number of jumps > 2 semitones
            - octave_errors: number of jumps > 8 semitones (likely octave errors)
            - max_semitone_jump: largest semitone jump detected
    """
    if len(pitch_values) < 2:
        return 0, 0, 0.0

    # Convert Hz to semitones for analysis
    # Semitone difference = 12 * log2(f2/f1)
    semitone_diffs = []

    for i in range(1, len(pitch_values)):
        if pitch_values[i] > 0 and pitch_values[i - 1] > 0:  # Both frames voiced
            # Calculate semitone difference
            ratio = pitch_values[i] / pitch_values[i - 1]
            if ratio > 0:
                semitone_diff = abs(12 * np.log2(ratio))
                semitone_diffs.append(semitone_diff)

    if not semitone_diffs:
        return 0, 0, 0.0

    semitone_diffs = np.array(semitone_diffs)

    # Count jumps based on research-backed thresholds:
    # > 2 semitones: potentially unnatural (large jump threshold)
    # > 8 semitones: likely octave error or catastrophic failure
    semitone_jumps = int(np.sum(semitone_diffs > 2.0))
    octave_errors = int(np.sum(semitone_diffs > 8.0))
    max_semitone_jump = float(np.max(semitone_diffs))

    return semitone_jumps, octave_errors, max_semitone_jump
