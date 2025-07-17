import json
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from audioevals.utils.audio import AudioData
from audiobox_aesthetics.infer import initialize_predictor
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))


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
            "average_scores": {
                "CE": float,  # Content Enjoyment
                "CU": float,  # Content Usefulness
                "PC": float,  # Production Complexity
                "PQ": float   # Production Quality
            },
            "results": [
                {
                    "file_index": int,
                    "audio_file": str,
                    "CE": float,
                    "CU": float,
                    "PC": float,
                    "PQ": float,
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
        "average_scores": {"CE": 0.0, "CU": 0.0, "PC": 0.0, "PQ": 0.0},
        "results": [],
    }

    print("Starting Audiobox Aesthetics evaluation...")
    print(f"Audio directory: {audio_path}")
    print(f"Total files to evaluate: {len(ground_truth_transcripts)}")

    # Initialize the predictor
    try:
        predictor = initialize_predictor()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {str(e)}")
        raise

    print("-" * 50)

    batch_files = []
    audio_filenames = []

    for audio_filename, _ in ground_truth_transcripts.items():
        audio_file = audio_path / audio_filename

        if audio_file.exists():
            batch_files.append({"path": str(audio_file.resolve())})
            audio_filenames.append(audio_filename)
        else:
            print(f"⚠️  Audio file not found: {audio_file}")
            results["failed_evaluations"] += 1

    if not batch_files:
        print("❌ No audio files found to evaluate")
        return results

    print(f"Running batch prediction on {len(batch_files)} files...")

    try:
        # Run batch prediction
        predictions = predictor.forward(batch_files)

        all_scores = {"CE": [], "CU": [], "PC": [], "PQ": []}

        for i, prediction in enumerate(predictions):
            audio_filename = audio_filenames[i]
            audio_file = batch_files[i]["path"]

            result = {
                "audio_file": audio_file,
                "CE": prediction["CE"],
                "CU": prediction["CU"],
                "PC": prediction["PC"],
                "PQ": prediction["PQ"],
            }

            results["results"].append(result)
            results["successful_evaluations"] += 1

            # Collect scores for averaging
            for metric in ["CE", "CU", "PC", "PQ"]:
                all_scores[metric].append(prediction[metric])

            print(
                f"✅ {audio_filename} - CE: {prediction['CE']:.2f}, CU: {prediction['CU']:.2f}, PC: {prediction['PC']:.2f}, PQ: {prediction['PQ']:.2f}"
            )

        # Calculate averages
        for metric in ["CE", "CU", "PC", "PQ"]:
            if all_scores[metric]:
                results["average_scores"][metric] = sum(all_scores[metric]) / len(
                    all_scores[metric]
                )

    except Exception as e:
        print(f"❌ Batch prediction failed: {str(e)}")
        results["failed_evaluations"] = len(audio_filenames)

    results["results"].sort(key=lambda x: x["audio_file"])

    print("-" * 50)
    print("Audiobox Aesthetics Evaluation Complete!")
    print(
        f"✅ Successfully evaluated: {results['successful_evaluations']}/{results['total_files']}"
    )
    print(f"❌ Failed evaluations: {results['failed_evaluations']}")

    if results["successful_evaluations"] > 0:
        print("\n📊 Average Scores:")
        print(f"   CE (Content Enjoyment): {results['average_scores']['CE']:.2f}")
        print(f"   CU (Content Usefulness): {results['average_scores']['CU']:.2f}")
        print(f"   PC (Production Complexity): {results['average_scores']['PC']:.2f}")
        print(f"   PQ (Production Quality): {results['average_scores']['PQ']:.2f}")

    print("-" * 50)

    return results


def run_single_file(audio_file_path: str) -> Dict:
    """
    Returns:
        {
            "CE": float,
            "CU": float,
            "PC": float,
            "PQ": float,
        }
    """
    audio_data = AudioData.from_wav_file(audio_file_path)
    return run_audio_data(audio_data)


def run_audio_data(audio_data: AudioData) -> Dict:
    """
    Returns:
        {
            "CE": float,
            "CU": float,
            "PC": float,
            "PQ": float,
        }
    """
    result = {
        "CE": 0.0,
        "CU": 0.0,
        "PC": 0.0,
        "PQ": 0.0,
    }
    
    # Initialize the predictor
    predictor = initialize_predictor()
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_data.get_1d_array(dtype=np.float32)).unsqueeze(0)
    
    prediction = predictor.forward([{"path": audio_tensor, "sample_rate": audio_data.sample_rate}])[0]
    
    result["CE"] = prediction["CE"]
    result["CU"] = prediction["CU"]
    result["PC"] = prediction["PC"]
    result["PQ"] = prediction["PQ"]
    
    return result