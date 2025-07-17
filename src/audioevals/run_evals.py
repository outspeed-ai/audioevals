import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from audioevals.utils.common import normalize_text
from audioevals.evals import audiobox_eval, wer_eval, vad_eval

EVAL_TYPES = ["wer", "audiobox", "vad"]


async def run_evaluations(
    transcripts_file: Optional[str] = None,
    dataset: Optional[str] = None,
    eval_types: Optional[List[str]] = None,
) -> Dict:
    if eval_types is None:
        eval_types = EVAL_TYPES

    output_path = Path(dataset)
    audio_dir = output_path / "audios"

    if transcripts_file is None:
        transcripts_path = output_path / "transcripts.json"
    else:
        transcripts_path = Path(transcripts_file)

    with open(transcripts_path, "r") as f:
        ground_truth_transcripts = json.load(f)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "audio_directory": str(Path(audio_dir).resolve()),
            "transcripts_file": str(transcripts_path.resolve()),
            "evaluation_types": eval_types,
        },
        "evaluations": [],
    }

    print(f"{'=' * 60}")
    print("🎯 Running TTS evaluations")
    print(f"{'=' * 60}")

    for audio_filename, ground_truth in ground_truth_transcripts.items():
        audio_file = audio_dir / audio_filename

        eval_item = {
            "audio_file": str(audio_file),
            "ground_truth": normalize_text(ground_truth),
        }
        results["evaluations"].append(eval_item)

    for eval_type in eval_types:
        try:
            print(f"\n🔍 Running {eval_type.upper()} evaluation...")

            if eval_type == "wer":
                eval_results = await wer_eval.run(
                    audio_dir=str(audio_dir), transcripts_file=str(transcripts_path)
                )

                for result in eval_results["results"]:
                    audio_file = result["audio_file"]
                    # Find matching evaluation item
                    for eval_item in results["evaluations"]:
                        if eval_item["audio_file"] == audio_file:
                            eval_item["stt_transcript"] = result["stt_transcript"]
                            eval_item["wer"] = result["wer_score"]
                            break

            elif eval_type == "audiobox":
                eval_results = audiobox_eval.run(
                    audio_dir=str(audio_dir), transcripts_file=str(transcripts_path)
                )

                for result in eval_results["results"]:
                    audio_file = result["audio_file"]

                    for eval_item in results["evaluations"]:
                        if eval_item["audio_file"] == audio_file:
                            eval_item["audiobox"] = {
                                "CE": result["CE"],
                                "CU": result["CU"],
                                "PC": result["PC"],
                                "PQ": result["PQ"],
                            }
                            break

            elif eval_type == "vad":
                eval_results = vad_eval.run(
                    audio_dir=str(audio_dir), transcripts_file=str(transcripts_path)
                )

                for result in eval_results["results"]:
                    audio_file = result["audio_file"]

                    for eval_item in results["evaluations"]:
                        if eval_item["audio_file"] == audio_file:
                            eval_item["vad"] = {
                                "max_silence_duration": result["max_silence_duration"],
                                "silence_to_speech_ratio": result["silence_to_speech_ratio"],
                            }
                            break
            else:
                print(f"⚠️  Unknown evaluation type: {eval_type}")
                continue

        except Exception as e:
            print(f"❌ Error running {eval_type} evaluation: {str(e)}")
            continue

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    total_files = len(results["evaluations"])
    print(f"📁 Total files evaluated: {total_files}")

    if "wer" in eval_types:
        wer_scores = [
            item["wer"] for item in results["evaluations"] if "wer" in item
        ]
        if wer_scores:
            avg_wer = sum(wer_scores) / len(wer_scores)
            print(
                f"🎯 WER: Average {avg_wer:.2f}%, Range {min(wer_scores):.2f}%-{max(wer_scores):.2f}%"
            )

    if "audiobox" in eval_types:
        audiobox_results = [
            item.get("audiobox", {})
            for item in results["evaluations"]
            if "audiobox" in item
        ]
        if audiobox_results:
            avg_pq = sum(r.get("PQ", 0) for r in audiobox_results) / len(
                audiobox_results
            )
            print(f"🎵 Audiobox: Average PQ {avg_pq:.2f}")

    if "vad" in eval_types:
        vad_results = [
            item.get("vad", {})
            for item in results["evaluations"]
            if "vad" in item
        ]
        if vad_results:
            avg_max_silence = sum(r.get("max_silence_duration", 0) for r in vad_results) / len(vad_results)
            avg_silence_ratio = sum(r.get("silence_to_speech_ratio", 0) for r in vad_results) / len(vad_results)
            print(f"🔇 VAD: Average max silence {avg_max_silence:.2f}s, Silence/Speech ratio {avg_silence_ratio:.2f}")

    print(f"\n💾 Results saved to: {results_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run TTS evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--transcripts",
        help="Path to transcripts.json file (default: it expects transcripts.json in the dataset directory)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=EVAL_TYPES,
        choices=EVAL_TYPES,
        help="Types of evaluations to run (default: all)",
    )

    args = parser.parse_args()

    # Run evaluations
    asyncio.run(run_evaluations(
        transcripts_file=args.transcripts,
        dataset=args.dataset,
        eval_types=args.evals,
    ))

    sys.exit(0)


if __name__ == "__main__":
    main()
